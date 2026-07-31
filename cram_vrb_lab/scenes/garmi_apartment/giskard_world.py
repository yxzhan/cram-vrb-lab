"""Giskard world config that adds the garmi-apartment environment next to the
Stretch robot, so the shared semantic_digital_twin world giskard plans in also
contains the surroundings (walls, furniture) and can avoid collisions with them.

Purely additive: no changes to giskardpy or semantic_digital_twin. It subclasses
the stock ``WorldWithStretchConfigDiffDrive`` and, after the robot is built, merges
``scene-bodies.xml`` into the same world at the pose the apartment occupies in the
Isaac Sim scene (:mod:`cram_vrb_lab.scenes.garmi_apartment.isaac_scene`).

The counterpart for the other flat is
:mod:`cram_vrb_lab.scenes.apartment.giskard_world`; the two differ in exactly two
ways, both because this environment is an **MJCF converted from the very USD Isaac
renders** rather than an independently authored URDF:

- **No mesh dropping.** ``apartment.urdf`` points at meshes in ROS packages that
  may not be installed, so its loader strips unresolvable geometry and keeps only
  primitive collision shapes. This MJCF resolves its meshes through ``meshdir`` /
  ``texturedir`` relative to the file itself, so it parses whole and giskard gets
  the full-resolution environment.
- **No measured alignment.** The URDF needed a probed 180 deg yaw plus an x/y/z
  offset; here the transform is the identity. See
  :mod:`cram_vrb_lab.scenes.garmi_apartment.constants`.

.. note::
   15 of the 79 bodies -- the floor lamp, the dining table, the two chairs and the
   eleven books -- come out of the MJCF on ``Connection6DoF`` connections, because
   they are free bodies in MuJoCo. They are left that way. ``Connection6DoF`` is
   documented as having degrees of freedom "that cannot be actively controlled", so
   they add no variables to giskard's QP and no goal can drive them; their poses
   come from the MJCF and nothing moves them. The remaining articulation -- three
   room doors, four drawers and four cabinet doors -- parses to the revolute and
   prismatic connections giskard expects.

.. warning::
   Those free bodies make the **merge order load-bearing**, which is the one real
   hazard in this module. ``StretchRealStyleInterface.setup`` identifies the
   ``map -> odom`` localization joint as
   ``world.get_connections_by_type(Connection6DoF)[0]`` -- by position, not by name
   -- and this world contains 16 of them rather than the usual 1. ``[0]`` is the
   right one only because ``WorldWithDiffDriveRobot.setup_world`` creates the
   localization connection before anything else and :meth:`setup_world` below
   merges the environment *after* calling ``super()``. Verified on the assembled
   world: ``get_connections_by_type(Connection6DoF)[0]`` is ``map -> odom`` and is
   the same object as the config's ``localization`` field. Merge the environment
   first and giskard would sync a book to the robot's localization transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    WorldWithStretchConfigDiffDrive,
)
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection

from .constants import GARMI_APARTMENT_MJCF_PATH, garmi_apartment_pose_in_map


def load_garmi_apartment_mjcf(mjcf_path: str = GARMI_APARTMENT_MJCF_PATH) -> World:
    """Parse the apartment MJCF into its own :class:`World`.

    Returns the parsed world rather than a description string -- the asymmetry with
    ``load_apartment_urdf`` is just that ``URDFParser`` takes a URDF *string* it can
    pre-edit, while ``MJCFParser`` goes through MuJoCo's own compiler and needs the
    file on disk (its ``meshdir`` / ``texturedir`` resolve relative to it).
    """
    return MJCFParser(file_path=mjcf_path).parse()


@dataclass
class WorldWithStretchAndGarmiApartmentDiffDrive(WorldWithStretchConfigDiffDrive):
    """Stretch (diff-drive) plus the garmi-apartment environment in one giskard world."""

    apartment_pose: HomogeneousTransformationMatrix = field(
        kw_only=True, default_factory=garmi_apartment_pose_in_map
    )
    """MJCF root pose in the ``map`` frame; see
    :func:`~cram_vrb_lab.scenes.garmi_apartment.constants.garmi_apartment_pose_in_map`."""

    mjcf_path: str = field(kw_only=True, default=GARMI_APARTMENT_MJCF_PATH)
    """Environment description merged next to the robot."""

    def setup_world(self) -> None:
        """Build the robot world as usual, then merge the apartment onto ``map``.

        Runs inside the ``modify_world`` context that :class:`giskardpy.middleware.
        ros2.giskard.Giskard` already opens around ``setup_world`` (matching the
        base class, which likewise assumes that outer context).
        """
        super().setup_world()

        apartment_world = load_garmi_apartment_mjcf(self.mjcf_path)
        map_to_apartment = FixedConnection(
            parent=self.world.root,
            child=apartment_world.root,
            parent_T_connection_expression=self.apartment_pose,
        )
        self.world.merge_world(apartment_world, map_to_apartment)
