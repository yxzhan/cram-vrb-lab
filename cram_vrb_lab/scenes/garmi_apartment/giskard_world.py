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

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field

from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    WorldWithStretchConfigDiffDrive,
)
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)

from .constants import GARMI_APARTMENT_MJCF_PATH, garmi_apartment_pose_in_map

REVOLUTE_VELOCITY_LIMIT = math.pi / 2
"""Velocity limit [rad/s] given to the MJCF's doors and cabinet doors.

Not invented: this is the value every revolute environment joint in
``iai_apartment/urdf/apartment.urdf`` carries (``velocity="1.5708"``), so the two
scenes present giskard with furniture that swings at the same speed.
"""

PRISMATIC_VELOCITY_LIMIT = 0.5
"""Velocity limit [m/s] given to the MJCF's drawers.

The value the apartment URDF uses for every drawer (``velocity="0.5"``).
"""


def _mjcf_angles_are_degrees(mjcf_path: str) -> bool:
    """Whether the MJCF's ``<compiler angle=...>`` is ``degree`` (MuJoCo's default).

    Read from the file rather than assumed, so a future re-export in radians stops
    the conversion in :func:`_repair_environment_dof_limits` from being applied twice.
    """
    compiler = ElementTree.parse(mjcf_path).getroot().find("compiler")
    if compiler is None:
        return True  # MuJoCo's own default when the attribute is absent
    return compiler.get("angle", "degree") == "degree"


def _repair_environment_dof_limits(world: World, angles_in_degrees: bool) -> None:
    """Fill in the joint limits ``MJCFParser`` cannot supply, in place.

    Two separate defects, both fatal in their own way, and both a consequence of MJCF
    simply not carrying what a motion planner needs:

    1. **No velocity limits at all.** ``MJCFParser.parse_dof`` sets only ``position``
       from the MuJoCo ``range``; ``velocity`` stays ``None``. Giskard builds a
       velocity-convergence term for *every* goal
       (``graph_node.velocity_convergence_expression``), which evaluates
       ``dof.limits.upper.velocity * joint_convergence_threshold`` over
       ``world.active_degrees_of_freedom``. With a ``None`` in there, **any** command
       -- a look, a drive, a reach -- dies with ``TypeError: unsupported operand
       type(s) for *: 'NoneType' and 'float'`` before planning starts. The eleven
       articulated environment joints are active DOFs, so this scene could not
       execute anything at all.
    2. **Revolute position limits in degrees.** This MJCF declares
       ``<compiler angle="degree">`` and writes its hinge ranges accordingly
       (``door_2_leaf`` is ``range="0 180"``), but the parser reads
       ``mujoco_joint.range`` straight off the spec, before MuJoCo's compiler would
       normalise it. Giskard reads those as **radians**, so a door that should open
       180 deg is modelled as free to spin ~28 turns. Slide joints are lengths and
       are left alone.

    Only revolute and prismatic connections are touched, so the robot's own DOFs --
    parsed from URDF, already complete -- and the 15 passive ``Connection6DoF`` free
    bodies are never modified.
    """
    for connection in world.connections:
        if isinstance(connection, RevoluteConnection):
            velocity_limit = REVOLUTE_VELOCITY_LIMIT
            rescale = math.radians if angles_in_degrees else None
        elif isinstance(connection, PrismaticConnection):
            velocity_limit = PRISMATIC_VELOCITY_LIMIT
            rescale = None  # a drawer's range is a length, not an angle
        else:
            continue

        for dof in connection.active_dofs:
            limits = dof.limits
            if limits is None:
                # No ``range`` in the MJCF: an unlimited joint. Giskard still needs a
                # velocity, but inventing a position range would invent a constraint.
                continue
            for bound, sign in ((limits.lower, -1.0), (limits.upper, 1.0)):
                if rescale is not None and bound.position is not None:
                    bound.position = float(rescale(bound.position))
                if bound.velocity is None:
                    bound.velocity = sign * velocity_limit


def load_garmi_apartment_mjcf(mjcf_path: str = GARMI_APARTMENT_MJCF_PATH) -> World:
    """Parse the apartment MJCF into its own :class:`World`, with joint limits repaired.

    Returns the parsed world rather than a description string -- the asymmetry with
    ``load_apartment_urdf`` is just that ``URDFParser`` takes a URDF *string* it can
    pre-edit, while ``MJCFParser`` goes through MuJoCo's own compiler and needs the
    file on disk (its ``meshdir`` / ``texturedir`` resolve relative to it). The
    post-processing step is the counterpart of that function's mesh dropping: the
    minimum edit that makes the description usable, applied here rather than upstream.
    See :func:`_repair_environment_dof_limits`.
    """
    world = MJCFParser(file_path=mjcf_path).parse()
    _repair_environment_dof_limits(world, _mjcf_angles_are_degrees(mjcf_path))
    return world


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
