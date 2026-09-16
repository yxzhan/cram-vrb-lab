"""Make a scene authored for PhysX loadable by Newton.

PhysX simulates a joint between two rigid bodies wherever it finds one. Newton
does not: every joint has to belong to an articulation, and one that does not
fails the whole stage ::

    [Newton] Initialization failed: Found 11 joint(s) not belonging to any
    articulation. Call add_articulation() for all joints. Orphan joints:
    .../door/Actor_0000/HingeJoint, .../cabinet/Actor_0000/joints/door_1_hinge

The failure is not confined to the joints it names -- Newton then builds no model
at all, so there is no physics view, and every robot in the scene is a dead
handle whose ``dof_names`` is ``None``.

The apartment's doors and the kitchen cabinet are exactly this: an asset root
holding a kinematic frame, the leaves that swing off it, and the hinges between
them, with no articulation root anywhere. :func:`articulate_loose_joints` puts one
on each such asset, which is what the asset would carry had it been authored for
an articulated engine.

The scenery around them fails the other half of the same rule ::

    [Newton] Initialization failed: SolverMuJoCo cannot convert bodies that are
    outside articulations and have no standalone joint to world.
    Bodies: .../Base/Floor/Actor_0000/Geom/A92, ...

-- the floor, the walls and the furniture are authored as *kinematic* rigid
bodies, and MuJoCo has nowhere to put a body that is neither in an articulation
nor jointed to the world. They are scenery: nothing simulates them, they are only
there to be collided with. :func:`staticize_kinematic_bodies` says so, by
disabling the rigid body and leaving the colliders, which Newton then attaches
to the world.

:func:`prepare_scene` runs both, and is a no-op on a scene that needs neither.
"""

from pxr import Usd, UsdPhysics


def prepare_scene(stage, root_path="/World"):
    """Make everything under *root_path* something Newton can build a model from.

    Order matters: the articulation roots go on first, so the kinematic bodies
    they take in -- a door's frame, the cabinet's carcase -- are recognised as
    part of an articulation and left alone.

    :return: ``(articulated, staticized)``, the prim paths each pass changed.
    """
    articulated = articulate_loose_joints(stage, root_path)
    staticized = staticize_kinematic_bodies(stage, root_path)
    return articulated, staticized


def staticize_kinematic_bodies(stage, root_path="/World"):
    """Turn kinematic scenery under *root_path* into plain static collision geometry.

    A kinematic rigid body is one the simulator never moves, which is how these
    scenes say "scenery"; PhysX is happy to carry it as a body, Newton's MuJoCo
    solver is not. Disabling the body (rather than stripping the schema) is how
    the rest of this repo already spells static geometry -- see
    ``SceneResetROS._is_enabled`` -- and Newton's importer skips a disabled body
    and attaches its shapes to the world.

    Bodies inside an articulation and bodies a joint connects are left alone:
    those are doors and drawers, which are meant to move.

    :return: the prim paths it disabled.
    """
    jointed = set()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            for relationship in (joint.GetBody0Rel(), joint.GetBody1Rel()):
                jointed.update(str(target) for target in relationship.GetTargets())

    disabled = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if str(prim.GetPath()) in jointed or _articulation_root_above(prim, inclusive=True):
            continue
        body = UsdPhysics.RigidBodyAPI(prim)
        kinematic = body.GetKinematicEnabledAttr().Get()
        enabled = body.GetRigidBodyEnabledAttr().Get()
        if kinematic and enabled is not False:
            body.CreateRigidBodyEnabledAttr().Set(False)
            disabled.append(prim.GetPath().pathString)
    return disabled


def articulate_loose_joints(stage, root_path="/World"):
    """Apply ``ArticulationRootAPI`` to every asset under *root_path* that owns a
    joint but sits in no articulation. Returns the prim paths it marked.

    The root chosen for a joint is the common ancestor of the joint and both of
    its bodies -- the asset, not the joint's own parent, since the bodies it
    connects are the joint's siblings or cousins. Several joints of one asset
    (the cabinet has six) resolve to the same root and mark it once.
    """
    marked = []
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return marked

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        if _articulation_root_above(prim):
            continue
        asset = _common_ancestor(prim, UsdPhysics.Joint(prim))
        if asset is None or _articulation_root_above(asset, inclusive=True):
            continue
        UsdPhysics.ArticulationRootAPI.Apply(asset)
        marked.append(asset.GetPath().pathString)
    return marked


def _articulation_root_above(prim, inclusive=False):
    """Whether *prim* is already inside (or is) an articulation."""
    current = prim if inclusive else prim.GetParent()
    while current.IsValid() and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.ArticulationRootAPI):
            return True
        current = current.GetParent()
    return False


def _common_ancestor(joint_prim, joint):
    """The shallowest prim containing the joint and both bodies it connects.

    A joint with only one body -- one end anchored to the world -- is answered by
    the ancestor it shares with that body.
    """
    paths = [joint_prim.GetPath()]
    for relationship in (joint.GetBody0Rel(), joint.GetBody1Rel()):
        paths += [target for target in relationship.GetTargets()]

    common = paths[0].GetParentPath()
    for path in paths[1:]:
        common = _common_prefix(common, path)
    prim = joint_prim.GetStage().GetPrimAtPath(common)
    return prim if prim.IsValid() and not prim.IsPseudoRoot() else None


def _common_prefix(a, b):
    """The deepest path both *a* and *b* are under (or are)."""
    a_names, b_names = a.GetPrefixes(), b.GetPrefixes()
    common = a_names[0]
    for left, right in zip(a_names, b_names):
        if left != right:
            break
        common = left
    return common
