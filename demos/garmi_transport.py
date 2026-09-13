"""Upstream's GARMI transport demonstration, pointed at the physics simulation.

``coraplex/demos/coraplex_garmi_demo/demo.py`` carries a bowl off the kitchen worktop
and a spoon out of a drawer, and places both on the table. It has only ever run as
``ExecutionType.SIMULATED`` -- a world built from the apartment's MJCF and GARMI's
URDF, stepped kinematically, with nothing on the network.

This file does not reimplement any of that. It **imports upstream's
:class:`GarmiApartmentDemonstration` and subclasses it**, so the scene, the objects
and the plan stay upstream's and keep tracking upstream as it changes. Everything
this repo needs to add lives in the subclass, and there is exactly one seam for it:
:meth:`GarmiApartmentOnIsaac.patch_before_plan`, called immediately before the plan
is built and performed.

``RobotDemonstration`` already knows how to talk to a live controller, so pointing it
at the sim is a constructor argument rather than a rewrite: ``ExecutionType.REAL``
makes ``acquire_world`` fetch the world from the running giskard server and keep it
synchronized instead of building its own.

What this file owns is therefore narrow: **the physics simulation and the viewer**.
Two things a kinematic run never needed, each its own patch on that seam:

- the objects have to exist in the render, at the poses the twin chose
  (:meth:`GarmiApartmentOnIsaac.sync_to_isaac`);
- and a grasped object has to be held by the hand in the render, since nothing about
  ``PickUpAction`` re-parenting it in the twin makes the render carry it
  (:meth:`GarmiApartmentOnIsaac.start_carry_sync`).

Where the gripper takes hold is *not* among them any more: an annotation says where it
may be grasped -- ``Bowl.grasp_poses`` traces its rim wall from the mesh, ``Cuttlery``
reaches down across the piece -- so a run against physics no longer needs the body frame
moved onto a hand-picked point the way it used to.

The viewer is :mod:`coraplex.visualization`'s ``cramera`` backend, started on the world
this demonstration acquired and given the plan it is about to perform, so the two show
one run.

Running it
==========

::

    binder/cram_python_wrapper.sh demos/garmi_transport.py

:func:`main` starts the Isaac scene and the giskard server itself and waits for both to
come up (the first Isaac start takes a few minutes: shader compilation). They are left
running afterwards, so the result stays on screen and a second run costs nothing --
``--no-launch`` uses whatever is already up, and ``--stop-after`` tears both down when
the plan ends. The equivalent by hand is::

    binder/isaacsim_python_wrapper.sh demos/sim.py --robot garmi --scene garmi_apartment --spawn-position 0.0 5.0 0.0259 --spawn-yaw 1.5707963267948966
    binder/cram_python_wrapper.sh demos/giskard_server.py --robot garmi --scene garmi_apartment --control-hz 15 --spawn-position 0.0 5.0 0.0259 --spawn-yaw 1.5707963267948966
    binder/cram_python_wrapper.sh demos/garmi_transport.py --no-launch

Nothing has to be spawned into the scene beforehand. Upstream's ``populate_scene``
puts the bowl and the spoon into the twin, and
:meth:`GarmiApartmentOnIsaac.sync_to_isaac` puts *those same two files* into the
render at the poses the twin already chose. That replaced a hardcoded prop table
the sim used to spawn at load time behind an ``ISAAC_TRANSPORT_PROPS`` flag: the
table had to state the poses a second time, and the two statements drifted apart
whenever either side moved. One description now, and it belongs to the plan.

``--simulated`` runs upstream's own kinematic path unchanged and needs neither
process, which is the reference to compare a real run against.

:mod:`demos.garmi_demo` is where these patches were worked out cell by cell; this file
is the same thing without the scaffolding.
"""


from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
DEMOS = Path(__file__).resolve().parent
for search_path in (REPO, DEMOS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

# Read by coraplex.visualization when the backend is chosen, so it has to be set before
# that import. setdefault, so a caller can ask for "rviz" or "none" from the outside.
os.environ.setdefault("CORAPLEX_VISUALIZATION", "cramera")
os.environ.setdefault("ISAAC_HEADLESS", "1")
os.environ.setdefault("ISAAC_LIVESTREAM", "1")
os.environ["ISAAC_WINDOW"] = "512x288"

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.plan_node import PlanNode
from coraplex.visualization import WorldVisualization
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color

from cram_vrb_lab.sim.scene_sync import SceneSyncClient, shape_pose_in_world
from launcher import start_giskard_server, start_isaac_sim, stop, start_streaming_client
from cram_vrb_lab.sim.isaac_app import livestream_enabled

try:
    from cramera.live.overlay import mark_overlay_bodies
except ImportError:  # the viewer is optional; without it there is nobody to tell

    def mark_overlay_bodies(*names: str) -> None:
        """No-op stand-in for a run without the cramera viewer installed."""


# Imported for its side effect. Upstream's build_context passes
# AlternativeMotion.discover_all(), which returns every AlternativeMotion subclass that
# has been *loaded*, so importing this module is what puts GARMI's three motions --
# the gripper and opening/closing a container, each carrying the CountSeconds deadline
# giskard has no equivalent of -- into upstream's context without overriding it.
import cram_vrb_lab.robots.garmi.motions  # noqa: F401

ROBOT, SCENE = "garmi", "garmi_apartment"
"""The setup :func:`main` launches; the pair has to be in ``cram_vrb_lab.setups.SETUPS``."""

SPAWN_POSITION = (0.0, 5.0, 0.0259)
SPAWN_YAW = math.pi / 2
"""Where GARMI starts, in ``map``. The sim and the giskard server are given the *same*
pose on purpose: it is where each believes the robot stands, and two different answers
means giskard plans for an arm that is not the one being rendered."""

CONTROL_HZ = 15
"""Rate the giskard QP loop runs at."""

SETTLE_SECONDS = 1.0
"""Seconds to let a freshly synced object come to rest before reading it back.

Both objects are placed where the twin put them, which is not where they end up: the
bowl is released above the worktop and the spoon inside a drawer, and each drops a
little. Pulling before they settle would write the falling pose into the twin and leave
the plan reaching for a place the object has already left.
"""

CARRY_SYNC_PERIOD = 0.1
"""Seconds between two looks at what the twin says the hand is holding."""

GARMI_PRIM_ROOT = "/garmi"
"""Prim the robot is spawned under, i.e. where a carried object is welded."""

UPSTREAM_DEMO_PATH = (
    REPO
    / "cognitive_robot_abstract_machine"
    / "coraplex"
    / "demos"
    / "coraplex_garmi_demo"
    / "demo.py"
)
"""Upstream's demonstration, which this file subclasses rather than copies."""


def load_upstream_demo() -> ModuleType:
    """Import :data:`UPSTREAM_DEMO_PATH` as a module.

    By path, because coraplex's demos are not part of its distribution: ``pyproject.toml``
    packages ``src/coraplex`` only, so ``coraplex_garmi_demo`` is not importable by name
    however coraplex itself was installed.

    Under an explicit module name rather than by putting the demo's directory on
    ``sys.path``, so that the very generic ``demo`` does not become an importable
    top-level module for everything else in the process.

    The module resolves ``iai_garmi_apartment`` through ``ament_index_python`` at import
    time, so the workspace has to be sourced -- which it is, for anything that can reach
    giskard at all.
    """
    spec = importlib.util.spec_from_file_location(
        "coraplex_garmi_demo_upstream", UPSTREAM_DEMO_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


upstream = load_upstream_demo()


@dataclass(frozen=True)
class IsaacObject:
    """How one of upstream's objects is rendered, weighed and grasped in the sim.

    Upstream decides *which* objects exist and where they start; this says what the
    physics simulation has to know on top of that, which is nothing a kinematic run
    ever needed.
    """

    name: str
    """The twin's name for the body, so upstream stays the one naming things."""

    mesh: str
    """The mesh file, upstream's own. **The same file the twin loaded**, not a copy:
    that is what makes the render and the plan describe one object."""

    mass: float
    """Mass [kg] for the render's rigid body."""

    color: Tuple[float, float, float]
    """RGB in 0..1, given to the twin's shape so the viewer agrees, and to the sim,
    which binds it as a material."""

    collider: str = "convexDecomposition"
    """``physics:approximation``. The decomposition holds the bowl's cavity open where a
    single hull would fill it in."""


ISAAC_OBJECTS = (
    IsaacObject(upstream.BOWL_NAME, upstream.BOWL_STL, 0.058, (0.20, 0.45, 0.80)),
    IsaacObject(upstream.SPOON_NAME, upstream.SPOON_STL, 0.05, (0.80, 0.80, 0.85)),
)
"""Upstream's two objects, with what the render needs on top of what the twin says."""

# The viewer streams a demo object's pose through its object overlay instead of baking it
# into the scene bundle it loads once, and decides which is which by name: this repo's own
# demos call their objects "bowl.stl", upstream calls them "bowl". Unregistered, these two
# land in the bundle, and re-parenting one onto the gripper changes the bundle's signature
# -- which the viewer answers by reloading the page, on every attach and every detach.
# Registered here, before upstream's populate_scene creates them, so the bundle is never
# built with them in it.
mark_overlay_bodies(*(scene_object.name for scene_object in ISAAC_OBJECTS))


@dataclass
class GarmiApartmentOnIsaac(upstream.GarmiApartmentDemonstration):
    """Upstream's demonstration with what a physics run and the viewer need.

    Inherits ``build_simulated_world``, ``is_scene_populated``, ``populate_scene``,
    ``build_context`` and the plan itself unchanged. The scene it builds for a
    ``--simulated`` run and the world this repo's giskard server serves are the same
    ``iai_garmi_apartment`` ``scene-bodies.xml``, so the two paths differ in how the
    robot is *driven*, never in what it is driven through.
    """

    visualization: Optional[WorldVisualization] = field(default=None, init=False)
    """The viewer, started once the world exists."""

    scene_sync: Optional[SceneSyncClient] = field(default=None, init=False)
    """Talks to the Isaac side. Lives on :attr:`isaac_node`, not on upstream's."""

    isaac_node: Optional[Node] = field(default=None, init=False)
    """This demo's own ROS node, spun by its own executor.

    Not upstream's session node, which a ``SingleThreadedExecutor`` spins: the carry
    timer below *blocks* on an ack that arrives on the same node, and a single-threaded
    executor cannot deliver it while a callback of its own is waiting for it. A second
    node with a multi-threaded executor keeps that deadlock out of upstream's session.
    """

    isaac_executor: Optional[MultiThreadedExecutor] = field(default=None, init=False)
    carry_timer: Any = field(default=None, init=False)
    _carry_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # %% seams

    def acquire_world(self) -> World:
        """Take upstream's world, then start the viewer on it.

        Here rather than in :meth:`build_plan`, so the viewer's first snapshot is the
        apartment as fetched rather than an empty world.
        """
        world = super().acquire_world()
        self.visualization = WorldVisualization.from_environment(world).start()
        return world

    def build_plan(self, context: Context) -> PlanNode:
        """Apply :meth:`patch_before_plan`, hand over to upstream, show the plan.

        ``RobotDemonstration.run`` calls this once per repetition, between populating
        the scene and performing the plan, which is the last point at which the world
        can still be adjusted.
        """
        self.patch_before_plan(context)
        plan = super().build_plan(context)
        if self.visualization is not None and plan.plan is not None:
            # The viewer lights the plan's nodes up as they run, so the timeline and the
            # motion are one run rather than two things to correlate by hand.
            self.visualization.attach_plan(plan.plan)
        return plan

    def patch_before_plan(self, context: Context) -> None:
        """Adjust the world for a physics run, immediately before the plan is built.

        Upstream's demonstration goes in unmodified until a real run shows what it
        cannot do, and each thing it cannot do earns one narrow, commented patch here
        rather than a fork of the plan. Everything below is idempotent: ``run`` calls
        this once per repetition.

        What a real run is still known to hit, unpatched:

        - **The lift does not move.** ``GarmiTorso`` drives both prismatic segments to
          one value, and in the sim only the upper one follows -- commanding
          ``TorsoState.MID`` leaves ``lift_0_lower_joint`` at 0.0 while
          ``lift_0_upper_joint`` reaches 0.2005. ``TorsoState.HIGH`` targets 0.4, which
          is exactly the upper limit of both joints' ``(0.0, 0.4)`` range, and aborts
          with ``InfeasibleException``. This one cannot be patched from here --
          ``TransportAction`` issues ``MoveTorsoAction(TorsoState.HIGH)`` itself, and it
          is a sim-side defect.
        - **A single exception ends the run.** ``RobotDemonstration.run`` has no retry
          anywhere, while :mod:`demos.garmi_demo` catches ``GiskardException`` and tries
          again. Against physics a transient infeasibility on the first motion is
          ordinary.

        :param context: The context the plan is about to be built against; its
            ``world`` is the one the controller is serving on a real run.
        """
        if self.execution_type is ExecutionType.SIMULATED:
            return
        self.sync_to_isaac(context.world)
        self.start_carry_sync(context.world)

    def tear_down(self) -> None:
        """Stop what this subclass started, then let upstream release its session."""
        if self.carry_timer is not None and self.isaac_node is not None:
            self.isaac_node.destroy_timer(self.carry_timer)
            self.carry_timer = None
        if self.isaac_executor is not None:
            self.isaac_executor.shutdown()
            self.isaac_executor = None
        if self.isaac_node is not None:
            self.isaac_node.destroy_node()
            self.isaac_node = None
        super().tear_down()

    # %% patches

    def sync_to_isaac(self, world: World) -> None:
        """Put the objects upstream spawned into the twin into Isaac as well.

        ``populate_scene`` is upstream's and spawns ``bowl.stl`` and ``spoon.stl``
        into the *twin*, which is all a kinematic run needs. A run against physics
        needs them in the render too, or the hand closes on nothing and
        ``PickUpAction`` attaches the body anyway, because a kinematic attach cannot
        fail.

        The same two files, at the poses the twin already put them at -- so the two
        sides describe one object rather than two descriptions someone has to keep
        in step.

        What crosses is the *mesh's* pose, not the body's: Isaac spawns a mesh file at
        the pose it is handed and knows nothing of the twin's body frames, so
        ``shape_pose_in_world`` composes any offset between the two back in (and ``pull``
        takes it out again on the way back). Upstream's objects carry no such offset, so
        this is the identity for them -- and it stays correct for an asset that does.

        ``track=True``: these are objects physics decides about. Once released they
        fall, settle, and get knocked around, and the twin should follow rather than
        keep asserting where the plan last put them. Tracking only makes the sim
        *publish*, though; ``pull`` is what writes those poses into the twin. This pulls
        once here, after letting the objects settle, so the plan starts from where they
        came to rest rather than from where they were released. A plan that wants to know
        again later calls ``pull`` again -- deliberately not automatic, because a pose
        applied mid-motion would move an object under the plan reaching for it.
        """
        sync = self.ensure_isaac_client()
        for scene_object in ISAAC_OBJECTS:
            body = world.get_body_by_name(scene_object.name)
            position, orientation = shape_pose_in_world(body)
            sync.place(
                scene_object.name,
                position,
                orientation,
                mesh=scene_object.mesh,
                collider=scene_object.collider,
                mass=scene_object.mass,
                color=scene_object.color,
                track=True,
            )
        print(f"  synced to Isaac: {sync.apply()}", flush=True)

        deadline = time.monotonic() + SETTLE_SECONDS
        while time.monotonic() < deadline:
            time.sleep(0.1)
        moved = sync.pull(world)
        print(
            "  pulled back: "
            + (", ".join(f"{n} {d * 1000:.1f} mm" for n, d in moved.items()) or "nothing"),
            flush=True,
        )

    def start_carry_sync(self, world: World) -> None:
        """Weld a grasped object to the hand in Isaac for as long as the twin holds it.

        A timer rather than a call after the pick, because ``TransportAction`` picks up,
        drives and puts down inside one ``perform()``, and nothing between those steps
        hands control back here. Each tick reads what the twin says -- ``PickUpAction``
        re-parents the body onto the tool frame and ``PlaceAction`` puts it back -- and
        sends the attach or detach that makes the render agree; a tick where nothing
        changed sends nothing at all.

        The lock is not politeness: ``apply`` blocks until the sim acknowledges, which
        outlasts the period by far, and the callback group below deliberately allows
        concurrent ticks, so without it several would be inside ``sync_attachments``
        (which mutates what is queued) at once. Skipping a tick is free -- the next one
        re-derives the same state from the world.
        """
        if self.carry_timer is not None:
            return
        sync = self.ensure_isaac_client()

        def tick() -> None:
            if not self._carry_lock.acquire(blocking=False):
                return
            try:
                if sync.sync_attachments(world, GARMI_PRIM_ROOT):
                    print(f"  carry: {sync.apply()}", flush=True)
            except Exception as failure:  # a demo must not die on the render
                print(f"  carry sync failed -- {type(failure).__name__}: {failure}",
                      flush=True)
            finally:
                self._carry_lock.release()

        # Its own callback group: apply() blocks on an ack that arrives on this node, and
        # rclpy puts a node's callbacks in one mutually exclusive group by default.
        self.carry_timer = self.isaac_node.create_timer(
            CARRY_SYNC_PERIOD, tick, callback_group=ReentrantCallbackGroup()
        )

    def ensure_isaac_client(self) -> SceneSyncClient:
        """The scene-sync client, on this demo's own spun node. See :attr:`isaac_node`."""
        if self.scene_sync is None:
            self.isaac_node = rclpy.create_node("garmi_transport_isaac")
            self.isaac_executor = MultiThreadedExecutor()
            self.isaac_executor.add_node(self.isaac_node)
            threading.Thread(
                target=self.isaac_executor.spin,
                daemon=True,
                name="garmi-transport-isaac-executor",
            ).start()
            self.scene_sync = SceneSyncClient(self.isaac_node)
        return self.scene_sync


def main(
    execution_type: ExecutionType = ExecutionType.REAL,
    launch: bool = True,
    stop_after: bool = False,
) -> None:
    """Run the demonstration, bringing up what it needs.

    :param execution_type: ``REAL`` drives the running giskard server and the Isaac
        scene behind it; ``SIMULATED`` is upstream's own kinematic path and needs
        neither, so nothing is launched for it.
    :param launch: whether to start the Isaac scene and the giskard server. Off when
        they are already up -- iterating on the plan does not want to pay for a sim
        start, which is minutes on a cold shader cache.
    :param stop_after: whether to tear both down when the plan ends. Off by default:
        the result stays on screen, and the next run starts immediately.
    """
    started = False
    if launch and execution_type is ExecutionType.REAL:
        # Both get the same spawn pose; see SPAWN_POSITION. The props are the four
        # kitchen objects the sim can put on the worktop at load time -- off here,
        # because this demonstration's objects are the ones upstream spawns into the
        # twin, and props the twin knows nothing about only get in the plan's way.
        start_isaac_sim(robot=ROBOT, scene=SCENE, spawn_position=SPAWN_POSITION, camera="none",
                        spawn_yaw=SPAWN_YAW, props=False)
        stream_proc = start_streaming_client() if livestream_enabled() else None
        start_giskard_server(robot=ROBOT, scene=SCENE, control_hz=CONTROL_HZ,
                             spawn_position=SPAWN_POSITION, spawn_yaw=SPAWN_YAW)
        started = True
    try:
        GarmiApartmentOnIsaac(
            used_robot=Garmi, execution_type=execution_type, collision_avoidance=False
        ).run()
    finally:
        if started and stop_after:
            stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="run upstream's kinematic path instead of driving the sim",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="use an Isaac scene and giskard server that are already running",
    )
    parser.add_argument(
        "--stop-after",
        action="store_true",
        help="stop the sim and the server when the plan ends",
    )
    arguments = parser.parse_args()
    main(
        ExecutionType.SIMULATED if arguments.simulated else ExecutionType.REAL,
        launch=not arguments.no_launch,
        stop_after=arguments.stop_after,
    )
