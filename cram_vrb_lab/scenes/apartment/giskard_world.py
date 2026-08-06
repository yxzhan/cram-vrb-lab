"""The apartment as giskard's world knows it, so the shared semantic_digital_twin
world giskard plans in also contains the surroundings (walls, furniture) and can
avoid collisions with them.

:func:`apartment_environment` describes ``apartment.urdf`` and the pose it occupies
in the Isaac Sim scene (:mod:`cram_vrb_lab.scenes.apartment.isaac_scene`);
:func:`cram_vrb_lab.control.giskard_world.build_world_config` merges it next to
whichever robot the demo runs, so no class here is specific to a robot.

.. warning::
   ``apartment.urdf`` references meshes from the ``iai_apartment`` / ``iai_kitchen``
   ROS packages. When those packages are not installed, the mesh geometry cannot
   be resolved and parsing would fail. ``load_apartment_urdf`` therefore drops
   every unresolvable mesh geometry and keeps only the primitive (box / cylinder /
   sphere) collision shapes already present in the URDF -- a coarse but real
   collision skeleton (walls and major furniture carry box collisions). Install
   the mesh packages to get the full-resolution environment.

.. note::
   ``apartment.urdf`` is a *different asset* from the ``apartmentICRA.usda`` scene
   rendered by Isaac Sim, and its root frame is flipped and offset relative to the
   USD origin. :func:`~cram_vrb_lab.scenes.apartment.constants.apartment_pose_in_map`
   encodes the measured alignment (180 deg yaw + offset) composed with Isaac's USD
   prim placement. Re-verify in RViz if the prim placement changes.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from typing_extensions import Optional

from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    PathResolver,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.exceptions import PathResolutionError
from semantic_digital_twin.world import World

from cram_vrb_lab.specs import EnvironmentSpec

from .constants import APARTMENT_URDF_PATH, apartment_pose_in_map


def _mesh_is_resolvable(filename: str, path_resolver: PathResolver) -> bool:
    """Return whether the mesh URI can be resolved to a local file."""
    try:
        path_resolver.resolve(filename)
        return True
    except PathResolutionError:
        return False


def load_apartment_urdf(
    urdf_path: str = APARTMENT_URDF_PATH,
    drop_unresolvable_meshes: bool = True,
    path_resolver: Optional[PathResolver] = None,
) -> str:
    """Read the apartment URDF, optionally dropping geometry whose mesh cannot be
    resolved so the description parses without the mesh packages installed.

    Only ``visual`` and ``collision`` elements backed by an unresolvable mesh are
    removed; primitive collision shapes (the walls and furniture boxes) are kept.
    """
    path_resolver = path_resolver or CompositePathResolver()
    with open(urdf_path) as urdf_file:
        tree = ElementTree.parse(urdf_file)

    if not drop_unresolvable_meshes:
        return ElementTree.tostring(tree.getroot(), encoding="unicode")

    for link in tree.getroot().findall("link"):
        for geometry_holder in list(link.findall("visual")) + list(
            link.findall("collision")
        ):
            mesh = geometry_holder.find("geometry/mesh")
            if mesh is None:
                continue
            if not _mesh_is_resolvable(mesh.get("filename"), path_resolver):
                link.remove(geometry_holder)

    return ElementTree.tostring(tree.getroot(), encoding="unicode")


def load_apartment_world(urdf_path: str = APARTMENT_URDF_PATH) -> World:
    """Parse the apartment URDF into its own world."""
    return URDFParser(urdf=load_apartment_urdf(urdf_path), prefix="").parse()


def apartment_environment() -> EnvironmentSpec:
    """The apartment as scenery for any robot's giskard world."""
    return EnvironmentSpec(load=load_apartment_world, pose=apartment_pose_in_map)
