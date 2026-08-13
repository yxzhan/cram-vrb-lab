"""The garmi-apartment as giskard's world knows it, so the shared
semantic_digital_twin world giskard plans in also contains the surroundings
(walls, furniture) and can avoid collisions with them.

:func:`garmi_apartment_environment` describes ``scene-bodies.xml`` and the pose it
occupies in the Isaac Sim scene
(:mod:`cram_vrb_lab.scenes.garmi_apartment.isaac_scene`);
:func:`cram_vrb_lab.control.giskard_world.build_world_config` merges it next to
whichever robot the demo runs, so no class here is specific to a robot.

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
   Those free bodies are why the **merge order is load-bearing**, the one real
   hazard this environment brings. ``StretchRealStyleInterface.setup`` identifies
   the ``map -> odom`` localization joint as
   ``world.get_connections_by_type(Connection6DoF)[0]`` -- by position, not by name
   -- and the assembled world contains 16 of them rather than the usual 1. ``[0]``
   is the right one only because ``WorldWithDiffDriveRobot.setup_world`` creates the
   localization connection before anything else and
   :class:`~cram_vrb_lab.control.giskard_world.WithEnvironment` merges the
   environment *after* calling ``super()``. Verified on the assembled world:
   ``get_connections_by_type(Connection6DoF)[0]`` is ``map -> odom`` and is the same
   object as the config's ``localization`` field. Merge the environment first and
   giskard would sync a book to the robot's localization transform.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
)

from cram_vrb_lab.specs import EnvironmentSpec

from .constants import GARMI_APARTMENT_MJCF_PATH, garmi_apartment_pose_in_map

REVOLUTE_VELOCITY_LIMIT = math.pi / 8
"""Velocity limit [rad/s] given to the MJCF's doors and cabinet doors.

Deliberately a quarter of what ``iai_apartment/urdf/apartment.urdf`` gives its
revolute joints (``velocity="1.5708"``), which is what this used to carry so that
the two flats' furniture moved alike. **What matters is not the hinge rate but
how fast the handle travels**: a cabinet-door handle sits about 0.41 m from its
hinge, so pi/2 rad/s drags it through 0.64 m/s -- faster than the drawers were,
and far faster than a gripper can hold on to. At pi/8 that is 0.16 m/s. Like
:data:`PRISMATIC_VELOCITY_LIMIT` this is a tuning value: multiply by the 0.41 m
lever arm to compare the two.
"""

PRISMATIC_VELOCITY_LIMIT = 0.1
"""Velocity limit [m/s] given to the MJCF's drawers.

Far below the ``velocity="0.5"`` the apartment URDF gives its drawers, which is
what this used to carry, and the reason both constants deviate from it.

This is the speed a drawer is *pulled out at*, and it is a property of the
furniture rather than of the robot: giskard's ``Open`` goal
(:class:`giskardpy.motion_statechart.goals.open_close.Open`) drives the
container's own DOF towards its limit, and the QP takes the tightest of that
DOF's velocity limit and the task's ``max_velocity`` (1.0 by default, and
:class:`~coraplex.robot_plans.actions.core.container.OpenAction` does not expose
it) -- so this constant, not anything on the arm, is what sets the pace.

At 0.5 m/s a 0.466 m drawer was out in under a second, which levered the handle
straight out from between the gripper's pads. Lowering it is half the fix; see
``FINGER_DRIVE_STIFFNESS`` in :mod:`cram_vrb_lab.robots.garmi.isaac_node` for the
other half. A tuning value, not a measured one -- a full drawer takes roughly
``0.466 / this`` seconds, so raise it once a grasp survives the pull.
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

    Both scenes hand giskard a parsed world (that is what an
    :class:`~cram_vrb_lab.specs.EnvironmentSpec` carries), but they get there
    differently: ``URDFParser`` takes a URDF *string* the apartment loader can
    pre-edit, while ``MJCFParser`` goes through MuJoCo's own compiler and needs the
    file on disk (its ``meshdir`` / ``texturedir`` resolve relative to it). The
    post-processing step below is the counterpart of that loader's mesh dropping: the
    minimum edit that makes the description usable, applied here rather than upstream.
    See :func:`_repair_environment_dof_limits`.
    """
    world = MJCFParser(file_path=mjcf_path).parse()
    _repair_environment_dof_limits(world, _mjcf_angles_are_degrees(mjcf_path))
    return world


def garmi_apartment_environment() -> EnvironmentSpec:
    """The garmi-apartment as scenery for any robot's giskard world."""
    return EnvironmentSpec(
        load=load_garmi_apartment_mjcf, pose=garmi_apartment_pose_in_map
    )
