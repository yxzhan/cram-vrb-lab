"""Giskard world config that adds the apartment environment next to the Stretch
robot, so the shared semantic_digital_twin world giskard plans in also contains
the surroundings (walls, furniture) and can avoid collisions with them.

Purely additive: no changes to giskardpy or semantic_digital_twin. It subclasses
the stock ``WorldWithStretchConfigDiffDrive`` and, after the robot is built,
merges ``apartment.urdf`` into the same world at the pose the apartment occupies
in the Isaac Sim scene (:mod:`cram_vrb_lab.scenes.apartment.isaac_scene`).

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
from dataclasses import dataclass, field
from typing_extensions import Optional

from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    WorldWithStretchConfigDiffDrive,
)
from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    PathResolver,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.exceptions import PathResolutionError
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import FixedConnection

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


@dataclass
class WorldWithStretchAndApartmentDiffDrive(WorldWithStretchConfigDiffDrive):
    """Stretch (diff-drive) plus the apartment environment in one giskard world."""

    apartment_urdf: str = field(kw_only=True, default_factory=load_apartment_urdf)
    """URDF string of the environment merged next to the robot."""

    apartment_pose: HomogeneousTransformationMatrix = field(
        kw_only=True, default_factory=apartment_pose_in_map
    )
    """Apartment root pose in the ``map`` frame; see :func:`apartment_pose_in_map`."""

    def setup_world(self) -> None:
        """Build the robot world as usual, then merge the apartment onto ``map``.

        Runs inside the ``modify_world`` context that :class:`giskardpy.middleware.
        ros2.giskard.Giskard` already opens around ``setup_world`` (matching the
        base class, which likewise assumes that outer context).
        """
        super().setup_world()

        apartment_world = URDFParser(urdf=self.apartment_urdf, prefix="").parse()
        map_to_apartment = FixedConnection(
            parent=self.world.root,
            child=apartment_world.root,
            parent_T_connection_expression=self.apartment_pose,
        )
        self.world.merge_world(apartment_world, map_to_apartment)
