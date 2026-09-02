"""Upstream's GARMI transport demonstration, run against the physics simulation.

``coraplex/demos/coraplex_garmi_demo/demo.py`` carries a bowl off the kitchen
worktop and a spoon out of a drawer, and places both on the table. It has only ever
run as ``ExecutionType.SIMULATED`` -- a world built from the apartment's MJCF and
GARMI's URDF, stepped kinematically, with nothing on the network. This is the same
demonstration wired to ``ExecutionType.REAL``: the world comes from the running
giskard server, the motions are executed by giskard, and the object being carried is
a *real rigid body* in Isaac Sim rather than a mesh that exists only in the twin.

Why this file rather than :mod:`demos.garmi_demo`
=================================================

:mod:`demos.garmi_demo` predates most of what CRAM now offers and hand-rolls it:
where to stand is a hardcoded ``STANDOFF`` per container, the furniture is annotated
one call at a time, and the carry is a list of ``MoveTCPWaypointsMotion`` poses. Each
of those has since grown a proper counterpart, and this file uses them:

============================  ==========================================
``garmi_demo.py``             here
============================  ==========================================
``STANDOFF = {Drawer: ...}``  ``reachability_location`` inside
                              ``TransportAction`` -- CRAM picks the pose
``annotate(Drawer, "...")``   ``WorldReasoner.infer_semantic_annotations``
``nudge_base(...)``           nothing; the navigation goal is resolved
manual pick/carry/place       one ``TransportAction``
============================  ==========================================

That matters beyond tidiness. The hardcoded ``STANDOFF[Drawer] = (1.05, 0.6)`` puts
the base 1.21 m from the handle, which needs ~0.96 m of shoulder-to-target reach --
about 84% of the arm's maximum extension, and past what it can hold a *commanded
tool orientation* at. ``reachability_location`` computes a standing pose instead of
asserting one, so the same plan keeps working when the park pose changes underneath
it (as it did upstream on 2026-08-29, which is what stranded the hardcoded number).

What deviates from upstream, and why
====================================

**The bowl is a real object.** Upstream spawns ``bowl.stl`` into the twin at a fixed
pose. Nothing in Isaac stands there, so a real run would close the hand on air and
``PickUpAction`` would attach the body to the gripper regardless -- a kinematic
attach cannot fail. This demonstration instead adopts the bowl that Isaac already
simulates: :data:`~cram_vrb_lab.scenes.garmi_apartment.constants.KITCHEN_PROPS`
puts ``bowl_left`` on the worktop with its own mass, inertia and convex-decomposed
collider, and :data:`BOWL_CENTRE_IN_MAP` is where the sim reported it *settled*.

**The grasp aims at the rim, not the centre.** See :data:`GRASP_ORIGIN_IN_BOWL` --
this is the one place the upstream plan cannot be reproduced verbatim, and the
reason is physical rather than incidental.

**No spoon.** Upstream's second transport takes a spoon out of ``drawer_1``. There is
no spoon in the Isaac scene -- ``assets/kitchen-objects`` holds a cereal box, a milk
box, a bowl and a cup, and no spoon asset exists to put in the drawer -- so carrying
one would be exactly the phantom-object run this file exists to avoid. The
drawer-opening branch of ``TransportAction`` (``inside_container`` ->
``_make_open_container_actions``) is therefore not exercised here; it needs a
graspable object physically inside a drawer first.

Running it
==========

Needs the Isaac scene and the giskard server already up, the scene started with
``ISAAC_KITCHEN_PROPS=1`` so the bowl is actually there::

    ISAAC_KITCHEN_PROPS=1 python demos/sim.py --robot garmi --scene garmi_apartment \\
        --spawn-position 0.0 5.0 0.0259 --spawn-yaw -1.5707963267948966
    python demos/giskard_server.py --robot garmi --scene garmi_apartment \\
        --control-hz 10 --spawn-position 0.0 5.0 0.0259 --spawn-yaw -1.5707963267948966
    python demos/garmi_transport_demo.py

``--simulated`` runs upstream's own kinematic path instead, against a world built
here from the same MJCF, and needs neither process.

.. warning::
   giskard's control loop has no deadline of its own
   (``giskardpy/middleware/ros2/control_loop.py``: ``while True: ...; if
   is_end_motion(): return``), so a motion whose end monitor never turns TRUE blocks
   the calling process for good. :data:`~cram_vrb_lab.robots.garmi.motions.GARMI_MOTION_MAPPINGS`
   puts a ``CountSeconds`` deadline on the three motions that are *known* to be able
   to stall -- the gripper and opening/closing a container -- but a Cartesian reach
   can stall too, and this plan is mostly Cartesian reaches. Run it where you can
   interrupt it.

Known blocker: the lift does not move
=====================================

**This plan cannot currently get past its second action**, and the reason is neither
CRAM's nor this file's. ``GarmiTorso`` drives both prismatic segments of the lift
column to the same value (``torso_low`` ``[0.0, 0.0]``, ``torso_mid``
``[0.2, 0.2]``, ``torso_high`` ``[0.4, 0.4]``), and in the sim only the *upper* one
follows. Read straight off ``/garmi/joint_states`` after commanding
``TorsoState.MID``::

    lift_0_lower_joint     +0.0000
    lift_0_upper_joint     +0.2005

So neither non-zero torso state can ever be reached, and the two fail differently:

- ``TorsoState.HIGH`` aborts with ``InfeasibleException: QP is infeasible`` after
  about 2.4 s. Its target is ``0.4``, which is *exactly* the upper limit of both
  joints' ``(0.0, 0.4)`` range -- see
  :func:`~cram_vrb_lab.robots.garmi.joints.joint_limits`.
- ``TorsoState.MID`` never returns at all: the upper segment arrives at 0.2, the
  lower stays at 0, the goal's monitor never turns TRUE and the deadline-free
  control loop above spins on it forever.

``TorsoState.LOW`` "works" only because it is ``[0.0, 0.0]`` and the lift already
rests there, so the goal is satisfied before it starts. That is the whole reason
:mod:`demos.garmi_demo` never tripped over this: ``reset_pos`` commands ``LOW`` and
nothing else in that file touches the torso.

There is no way to route around it from here. ``TransportAction`` issues
``MoveTorsoAction(TorsoState.HIGH)`` itself, between the pick-up and the navigate
for placing, and upstream's own comment records that ``CostmapLocation`` yields
nothing at a low torso -- so every CRAM transport plan needs a torso this robot
cannot currently raise. Fixing the lift in
:mod:`cram_vrb_lab.robots.garmi.isaac_node` is a prerequisite for this file, not a
detail of it; ``torso_high`` also wants to sit just inside the joint limit rather
than on it.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import ClassVar

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    ExecutionType,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.api import (
    BodySpecification,
    Connection6DoFSpecification,
    RobotSpecification,
    WorldSpecification,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale

from cram_vrb_lab.robots.garmi.motions import GARMI_MOTION_MAPPINGS
from cram_vrb_lab.scenes.garmi_apartment.constants import (
    DINING_TABLE_TOP,
    GARMI_APARTMENT_MJCF_PATH,
)

# %% the apartment and the robot in it

ODOM_T_GARMI_START = HomogeneousTransformationMatrix.from_xyz_rpy(
    0, 5, 0, yaw=-math.pi / 2
)
"""Where GARMI starts, in its ``odom`` frame -- only read by ``--simulated``.

The spawn pose :mod:`demos.garmi_demo` and the launcher use, rather than upstream's
``(0, 6, 0, yaw=+pi/2)``: a real run takes the base pose from ``/odom`` and this is
ignored, so the two paths should at least start the robot in the same place.
"""

DRIVE_TRANSLATION_VELOCITY_LIMITS = 0.1
"""How fast the base drives [m/s], as upstream sets it."""

DRIVE_ROTATION_VELOCITY_LIMITS = 0.1
"""How fast the base turns [rad/s], as upstream sets it."""

# %% the transported object

BOWL_NAME = "bowl"
"""Name of the transported bowl, which also marks whether the scene was populated."""

BOWL_CENTRE_IN_MAP = (-0.03, 7.23, 0.9783)
"""Centre of the ``bowl_left`` kitchen prop [m], in ``map``.

Not the pose it is *released* at (``KITCHEN_PROPS`` says ``(-0.03, 7.23, 0.945)``,
where 0.945 is the worktop surface) but where Isaac reports it **settled** after the
drop -- the sim prints ``Kitchen prop bowl_left: ... settled centre (-0.03, 7.23,
0.9783)`` at startup. The 33 mm between them is half the bowl's height, i.e. the
object standing on the worktop rather than hovering with its centre on it.

Re-read that line rather than trusting this constant if the worktop, the asset or
the drop height ever change.
"""

BOWL_SIZE = (0.1333, 0.1333, 0.0665)
"""Bounding box of the ``SM_SmallBowl`` asset [m], as the sim measures it.

The twin gets a box of these extents rather than the bowl mesh. What the twin needs
this shape for is collision checking and the grasp geometry below, and for both a
box that bounds the real collider is the honest approximation -- the alternative,
``coraplex/resources/objects/bowl.stl``, is a *different bowl* (0.1397 x 0.1390 x
0.0671) that happens to be nearly the same size, and pretending it is this one would
put the twin and the physics a few millimetres out of step for no gain.
"""

BOWL_WALL_THICKNESS = 0.006
"""Thickness of the bowl's rim [m], used to place :data:`GRASP_ORIGIN_IN_BOWL`."""

GRASP_ORIGIN_IN_BOWL = (
    -BOWL_SIZE[0] / 2 + BOWL_WALL_THICKNESS / 2,
    0.0,
    BOWL_SIZE[2] / 2,
)
"""Where the twin body's **origin** sits inside the bowl -- at the middle of its rim
wall, on the -x side, level with the top face.

This is the one deliberate departure from upstream's plan, and it is forced by the
hardware. ``PickUpAction`` aims the tool centre point at the object designator's
*origin* and then closes the hand; there is no offset parameter anywhere along that
path, so the origin **is** the grasp point. Upstream's bowl body carries its origin
at the bowl's centre, and a Franka Hand opens to at most 0.08 m
(:data:`~cram_vrb_lab.robots.garmi.joints.MAX_FINGER_TRAVEL` is 0.04 per finger)
while the bowl is 0.133 m across. Closing on the centre of a bowl half again wider
than the hand can open puts both fingers *inside* it and grips nothing.

A kinematic run never notices: ``PickUpAction`` attaches the body to the gripper
whatever the fingers did, so upstream's grasp looks fine in simulation and can only
fail once something has to actually hold the bowl up. Aiming at the rim instead
gives the fingers 6 mm of wall to close on, which is a grasp a parallel gripper can
physically make.

The shape is shifted back by the same vector (see :meth:`_bowl_specification`), so
moving the origin does not move the bowl: it still stands where Isaac put it.
"""

BOWL_TARGET_POINT = Point3.from_iterable(
    [DINING_TABLE_TOP[0] - 0.25, DINING_TABLE_TOP[1] + 0.42, DINING_TABLE_TOP[2]]
)
"""Where the bowl is carried to, in ``map``.

Upstream's ``(1.6, 5.2, 0.8)`` expressed against this scene's measured table instead
of as literals: :data:`~cram_vrb_lab.scenes.garmi_apartment.constants.DINING_TABLE_TOP`
is ``(1.85, 4.78, 0.771)``, the raycast height an object *rests* at, so this lands at
``(1.60, 5.20, 0.771)`` -- the same spot, on the near-left quarter of a top that runs
x in [1.433, 2.283], y in [4.101, 5.452].

The z is the table surface, not upstream's 0.8, because ``PlaceAction`` puts the
body's *origin* at this point and :data:`GRASP_ORIGIN_IN_BOWL` sits at the rim's top
-- half the bowl above its base. Asking for the surface therefore sets the bowl down
about a bowl-height too low; see the note in :meth:`build_plan`.
"""

# %% the demonstration


@dataclass
class GarmiApartmentTransport(RobotDemonstration):
    """GARMI carries the worktop bowl to the dining table.

    Upstream's :class:`GarmiApartmentDemonstration` with the spoon dropped and the
    bowl replaced by the one Isaac actually simulates; see the module docstring.
    """

    ros_node_name: ClassVar[str] = "garmi_transport_demo_node"

    def build_simulated_world(self) -> World:
        """Put GARMI into the apartment's MuJoCo scene, for a run with no controller.

        The same MJCF the giskard server merges into its own world
        (:data:`~cram_vrb_lab.scenes.garmi_apartment.constants.GARMI_APARTMENT_MJCF_PATH`
        -- ``iai_garmi_apartment``'s ``scene-bodies.xml``), so ``--simulated`` and a
        real run disagree about the robot's *state* but never about the scene.

        ``use_visual_as_collision_backup`` twice over: the scene keeps its collision
        geometry in a file that is not loaded, and GARMI's own shell -- the side,
        front and rear covers -- is drawn but never described for contact.
        """
        return WorldSpecification.from_mjcf(
            GARMI_APARTMENT_MJCF_PATH,
            use_visual_as_collision_backup=True,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    odom_T_robot_start=ODOM_T_GARMI_START,
                    drive_translation_velocity_limits=DRIVE_TRANSLATION_VELOCITY_LIMITS,
                    drive_rotation_velocity_limits=DRIVE_ROTATION_VELOCITY_LIMITS,
                )
            ],
        ).to_domain_object()

    def is_scene_populated(self, world: World) -> bool:
        """Whether the bowl is already in ``world``.

        A real run fetches a world that outlives this process, so re-running the
        demonstration against a server that is still up must not add a second bowl.
        """
        return world.is_kinematic_structure_entity_in_world_by_name(BOWL_NAME)

    def populate_scene(self, world: World) -> None:
        """Annotate the apartment's furniture, then add the bowl.

        The furniture first, as upstream does, so the reasoner describes the
        apartment the plan navigates rather than the object the plan already knows.
        On this scene that turns 145 bodies into 26 annotations -- 14 handles, 7
        doors, 4 drawers and a wardrobe -- in under two seconds, which is the whole
        of what :mod:`demos.garmi_demo` spells out by hand, one ``annotate`` call per
        container.
        """
        world_reasoner = WorldReasoner(world)
        inferred = world_reasoner.infer_semantic_annotations()
        with world.modify_world():
            world.add_semantic_annotations(inferred)

        # A 6DoF connection rather than the default fixed one: the bowl is picked up
        # and carried, and a pose can only be written to a connection that has the
        # degrees of freedom to carry it.
        Bowl.get_annotation_specification(
            BOWL_NAME,
            self._bowl_specification(),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world)

    @staticmethod
    def _bowl_specification() -> BodySpecification:
        """The twin's stand-in for Isaac's ``bowl_left`` prop.

        A box of the asset's measured extents, with the body frame at the rim
        (:data:`GRASP_ORIGIN_IN_BOWL`) and the shape shifted by the negative of that
        vector so the two cancel: the body's origin is the grasp point, and the box
        still stands where the sim settled it.
        """
        origin_in_bowl = HomogeneousTransformationMatrix.from_xyz_rpy(
            *(-value for value in GRASP_ORIGIN_IN_BOWL)
        )
        grasp_point_in_map = tuple(
            centre + offset
            for centre, offset in zip(BOWL_CENTRE_IN_MAP, GRASP_ORIGIN_IN_BOWL)
        )
        return BodySpecification.box(
            BOWL_NAME,
            Scale(*BOWL_SIZE),
            color=Color(0.85, 0.75, 0.25, 1.0),
            origin=origin_in_bowl,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                *grasp_point_in_map
            ),
        )

    def build_context(self, world: World) -> Context:
        """Build the plan context around the GARMI in ``world``.

        ``alternative_motion_mappings`` is this repo's GARMI set rather than
        upstream's ``AlternativeMotion.discover_all()``: those three motions
        (the gripper, opening and closing a container) are the ones that carry a
        ``CountSeconds`` deadline, without which a hand closing on a rigid object
        never reaches its commanded finger positions and the goal never ends.

        .. note:: The ROS node has to be in the context for a real robot.
        """
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            ros_node=self.ros_node,
            evaluate_conditions=True,
            alternative_motion_mappings=GARMI_MOTION_MAPPINGS,
        )

    def build_plan(self, context: Context) -> PlanNode:
        """Carry the bowl to the table.

        ``MoveTorsoAction(TorsoState.HIGH)`` leads, as upstream's own comment asks
        for: ``CostmapLocation`` yields no pose at all at a low torso, and
        ``next(iter(...))`` on the empty result is what fails.

        The grasp is upstream's -- approach from the right, aligned to the top,
        gripper rotated -- and only the point it aims at differs; see
        :data:`GRASP_ORIGIN_IN_BOWL`.

        The place target is the table surface. Because the body's origin sits at the
        top of the rim, that sets the bowl down roughly its own height too low, and
        the physics resolves the overlap by pushing it back out. Raising the target
        by ``BOWL_SIZE[2]`` would place it exactly, and is the first thing to try if
        the bowl ends up shoved across the table.
        """
        world = context.world
        end_effector = context.robot.get_right_arm_if_specified().end_effector

        return sequential(
            [
                ParkArmsAction(arm=Arms.BOTH),
                MoveTorsoAction(TorsoState.HIGH),
                TransportAction(
                    object_designator=world.get_semantic_annotations_by_type(Bowl)[0],
                    arm=Arms.RIGHT,
                    grasp_description=GraspDescription(
                        ApproachDirection.RIGHT,
                        VerticalAlignment.TOP,
                        end_effector,
                        rotate_gripper=True,
                    ),
                    target_location=Pose(
                        position=BOWL_TARGET_POINT, reference_frame=world.root
                    ),
                ),
            ],
            context,
        )


def main(execution_type: ExecutionType = ExecutionType.SIMULATED) -> None:
    """Run the demonstration.

    :param execution_type: ``REAL`` drives the running giskard server and the Isaac
        scene behind it; ``SIMULATED`` builds its own world and needs neither.
    """
    GarmiApartmentTransport(
        used_robot=Garmi, execution_type=execution_type, collision_avoidance=True
    ).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="run upstream's kinematic path instead of driving the sim",
    )
    arguments = parser.parse_args()
    main(
        ExecutionType.SIMULATED if arguments.simulated else ExecutionType.REAL
    )
