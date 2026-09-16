"""Import a URDF into the open stage, against the Isaac Sim 6.1 importer.

Isaac Sim 6.1 rewrote the URDF importer, and nothing of the 5.x call survives:

- the kit commands are gone. ``URDFParseAndImportFile`` no longer exists at all,
  and ``URDFCreateImportConfig`` only still exists as a deprecated shim in the
  *UI* extension. A command that is not registered does not raise --
  ``omni.kit.commands.execute`` returns ``(False, None)`` -- so the old call site
  died one line later, on the first attribute set against that ``None``.
- the importer no longer populates the open stage. ``URDFImporter.import_urdf``
  converts the URDF to a USD asset **on disk** and returns its path; putting the
  robot in the scene is now the caller's job, here a reference on *prim_path*.
- the generated asset carries a ``Physics`` variant set (``none``, ``physics``,
  ``physx``, ``mujoco``) with **no selection**. Referenced in as it comes, the
  robot has no joints, no articulation and no rigid bodies -- it is scenery.
  :func:`import_urdf_robot` selects ``physx``, which is what the PhysX backend
  the rest of this repo drives the robots through needs.
- the config fields were renamed and pruned: ``self_collision`` ->
  ``allow_self_collision``, ``convex_decomp`` -> ``collision_type``, and
  ``import_inertia_tensor`` / ``distance_scale`` are gone (the converter always
  uses the URDF's inertia tensors and always writes metres).

The link tree also comes out nested now, under ``<prim_path>/Geometry`` with the
joints beside it under ``<prim_path>/Physics``, where 5.x laid the links out flat
under the robot prim -- hence :func:`link_prim_path`, which looks a link up by
name instead of spelling a path out.
"""

import os
import tempfile

import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdPhysics

from cram_vrb_lab.sim.isaac_app import ensure_urdf_importer


def import_urdf_robot(
    urdf_text,
    prim_path,
    name,
    fix_base,
    merge_fixed_joints=False,
    self_collision=False,
    drive_type="acceleration",
):
    """Convert *urdf_text* to USD, reference it in at *prim_path*, and return the
    prim path of its articulation root (what ``Articulation`` wants).

    :param name: names both the generated asset and the robot's own prim inside
        it; only cosmetic, the reference lands on *prim_path* either way.
    :param fix_base: ``True`` welds the root link to the world, ``False`` leaves
        the robot floating. (The importer also takes ``None`` for "whatever the
        source says"; the robots here always mean one or the other.)
    :param merge_fixed_joints: off by default, as every robot here wants it: the
        TCP, mount and camera frames the semantic model looks bodies up by are
        fixed joints, and merging them away would leave the twin and the render
        describing different link trees.
    :param drive_type: ``"acceleration"`` keeps PhysX scaling each drive's force
        by the joint's effective inertia, which is what the gains in the robot
        modules are tuned against -- one number then serves a 20 kg lift column
        and a 1 kg head. The 5.x importer authored acceleration drives; the 6.1
        one reconstructs whatever the URDF implies, so it has to be asked.
    """
    ensure_urdf_importer()
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    # Kept for the process' lifetime, not cleaned up: the stage references the
    # generated USD, so it has to outlive the import (the meshes and textures the
    # converter copies next to it are read lazily, too).
    work_dir = tempfile.mkdtemp(prefix=f"{name}_urdf_import_")
    urdf_path = os.path.join(work_dir, f"{name}.urdf")
    with open(urdf_path, "w") as urdf_file:
        urdf_file.write(urdf_text)

    usd_path = URDFImporter(
        URDFImporterConfig(
            urdf_path=urdf_path,
            usd_path=work_dir,
            fix_base=fix_base,
            merge_fixed_joints=merge_fixed_joints,
            allow_self_collision=self_collision,
            # The collision meshes the descriptions ship are already convex per
            # link, so nothing needs decomposing.
            collision_type="Convex Hull",
            joint_drive_type=drive_type,
        )
    ).import_urdf()

    prim = add_reference_to_stage(usd_path, prim_path)
    prim.GetVariantSet("Physics").SetVariantSelection("physx")
    print(f"{name} converted to {usd_path} and referenced at {prim_path}")

    return articulation_root_path(prim_path)


def articulation_root_path(prim_path):
    """The prim under *prim_path* that carries ``ArticulationRootAPI``.

    5.x handed this back from the import command (``get_articulation_root=True``);
    6.1 has no equivalent, and the answer is not *prim_path* itself -- for a
    floating base it is the first link with mass, e.g. ``chassis_link`` for GARMI,
    whose massless ``base_link`` root becomes a plain Xform.
    """
    stage = omni.usd.get_context().get_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim.GetPath().pathString
    raise RuntimeError(f"no articulation root under {prim_path} after the import")


def link_prim_path(prim_path, link_name):
    """The prim of the link called *link_name* somewhere under *prim_path*.

    A name, not a path, because 6.1 nests the links as the link tree is nested:
    GARMI's head lives eleven levels down ``<prim_path>/Geometry``.

    A rigid body wins over a bare name match, because the meshes a link is drawn
    with are its children and repeat its name. A link that is *no* body is still
    found -- a massless frame on a fixed joint, such as the Stretch's camera frame,
    is a plain Xform here -- and being the shallowest match, pre-order finds it
    before anything hanging off it.
    """
    stage = omni.usd.get_context().get_stage()
    fallback = None
    for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
        if prim.GetName() != link_name:
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim.GetPath().pathString
        fallback = fallback or prim.GetPath().pathString
    if fallback:
        return fallback
    raise RuntimeError(f"no link named {link_name} under {prim_path}")


def urdf_link_names(urdf_text):
    """Every ``<link name=...>`` of a URDF, in document order."""
    from xml.etree import ElementTree

    return [link.get("name") for link in ElementTree.fromstring(urdf_text).findall("link")]


def collapsed_link_frames(prim_path, link_names, body_names):
    """The link frames the importer kept as plain Xforms instead of rigid bodies.

    A massless link on a fixed joint -- a TCP frame, an arm mount, ``link8`` -- is
    not a body of the 6.1 articulation (5.x gave every link one), so the physics
    view has no pose for it and TF published from that view loses the frame
    entirely. That is where the semantic model looks the tool frames up, so they
    have to come from somewhere; the prim is still in the imported USD, still in
    the right place, and a fixed joint means its offset from the nearest link that
    *is* a body never changes. So: measure it once, here, and publish it static.

    The massless *root* link (GARMI's ``base_link``) has no body above it and is
    skipped -- the robot's own odometry publishes that frame.

    :param link_names: the URDF's link names, to tell a link's prim from the mesh
        prims under it, which repeat its name.
    :param body_names: the articulation's bodies, i.e. the frames already covered.
    :return: ``(parent_body_name, link_name, translation, quaternion)`` per frame,
        the quaternion in ROS ``(x, y, z, w)`` order, ready for ``make_tf``.
    """
    import omni.usd as omni_usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    bodies = {str(name).split("/")[-1] for name in body_names}
    wanted = [name for name in link_names if name not in bodies]

    frames = []
    for name in wanted:
        prim = next(
            # Pre-order, so the link itself is found before the meshes under it.
            (p for p in Usd.PrimRange(root) if p.GetName() == name),
            None,
        )
        if prim is None:
            continue
        parent = prim.GetParent()
        while parent.IsValid() and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            parent = parent.GetParent()
        if not parent.IsValid() or not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            continue  # the massless root: nothing above it to hang the frame on
        # USD transforms compose as row vectors (p_world = p_local * M), so the
        # frame in its parent's coordinates is frame_to_world * world_to_parent.
        relative = omni_usd.get_world_transform_matrix(prim) * (
            omni_usd.get_world_transform_matrix(parent).GetInverse()
        )
        translation = relative.ExtractTranslation()
        rotation = relative.ExtractRotationQuat().GetNormalized()
        imaginary = rotation.GetImaginary()
        frames.append((
            parent.GetName(),
            name,
            [float(v) for v in translation],
            [float(imaginary[0]), float(imaginary[1]), float(imaginary[2]),
             float(rotation.GetReal())],
        ))
    return frames
