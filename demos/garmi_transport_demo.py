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

Running it
==========

Needs the Isaac scene and the giskard server already up::

    binder/isaacsim_python_wrapper.sh demos/sim.py --robot garmi --scene garmi_apartment --spawn-position 0.0 5.0 0.0259 --spawn-yaw 1.5707963267948966
    binder/cram_python_wrapper.sh demos/giskard_server.py --robot garmi --scene garmi_apartment --control-hz 15 --spawn-position 0.0 5.0 0.0259 --spawn-yaw 1.5707963267948966
    binder/cram_python_wrapper.sh demos/garmi_transport_demo.py

Nothing has to be spawned into the scene beforehand. Upstream's ``populate_scene``
puts the bowl and the spoon into the twin, and
:meth:`GarmiApartmentOnIsaac.sync_to_isaac` puts *those same two files* into the
render at the poses the twin already chose. That replaced a hardcoded prop table
the sim used to spawn at load time behind an ``ISAAC_TRANSPORT_PROPS`` flag: the
table had to state the poses a second time, and the two statements drifted apart
whenever either side moved. One description now, and it belongs to the plan.

``--simulated`` runs upstream's own kinematic path unchanged and needs neither
process, which is the reference to compare a real run against.
"""


from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.plan_node import PlanNode
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)

from cram_vrb_lab.sim.scene_sync import SceneSyncClient

# Imported for its side effect. Upstream's build_context passes
# AlternativeMotion.discover_all(), which returns every AlternativeMotion subclass that
# has been *loaded*, so importing this module is what puts GARMI's three motions --
# the gripper and opening/closing a container, each carrying the CountSeconds deadline
# giskard has no equivalent of -- into upstream's context without overriding it.
import cram_vrb_lab.robots.garmi.motions  # noqa: F401

SETTLE_SECONDS = 1.0
"""Seconds to let a freshly synced object come to rest before reading it back.

Both objects are placed where the twin put them, which is not where they end up:
``bowl.stl`` is released above the worktop and ``spoon.stl`` inside a drawer, and
each drops a little. Pulling before they settle would write the falling pose into
the twin and leave the plan reaching for a place the object has already left.
"""

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


@dataclass
class GarmiApartmentOnIsaac(upstream.GarmiApartmentDemonstration):
    """Upstream's demonstration with a hook for what a physics run needs.

    Inherits ``build_simulated_world``, ``is_scene_populated``, ``populate_scene``,
    ``build_context`` and the plan itself unchanged. The scene it builds for a
    ``--simulated`` run and the world this repo's giskard server serves are the same
    ``iai_garmi_apartment`` ``scene-bodies.xml``, so the two paths differ in how the
    robot is *driven*, never in what it is driven through.
    """

    def build_plan(self, context: Context) -> PlanNode:
        """Apply :meth:`patch_before_plan`, then hand over to upstream.

        ``RobotDemonstration.run`` calls this once per repetition, between populating
        the scene and performing the plan, which is the last point at which the world
        can still be adjusted.
        """
        self.patch_before_plan(context)
        return super().build_plan(context)

    def patch_before_plan(self, context: Context) -> None:
        """Adjust the world for a physics run, immediately before the plan is built.

        Upstream's demonstration goes in unmodified until a real run shows what it
        cannot do, and each thing it cannot do earns one narrow, commented patch here
        rather than a fork of the plan. So far there is one: :meth:`sync_to_isaac`.

        What a real run is already known to hit, still unpatched:

        - **The bowl is wider than the hand opens.** Upstream's mesh measures
          0.1397 x 0.1390 x 0.0671 m and its body origin sits at the centre;
          ``PickUpAction`` aims the tool centre point at that origin and closes, and a
          Franka Hand opens to 0.08 m
          (:data:`~cram_vrb_lab.robots.garmi.joints.MAX_FINGER_TRAVEL` is 0.04 per
          finger). Moving the origin to the rim is what makes it a grasp a parallel
          gripper can physically make.
        - **The lift does not move.** ``GarmiTorso`` drives both prismatic segments to
          one value, and in the sim only the upper one follows -- commanding
          ``TorsoState.MID`` leaves ``lift_0_lower_joint`` at 0.0 while
          ``lift_0_upper_joint`` reaches 0.2005. ``TorsoState.HIGH`` targets 0.4, which
          is exactly the upper limit of both joints' ``(0.0, 0.4)`` range, and aborts
          with ``InfeasibleException``; ``MID`` never converges at all. This one cannot
          be patched from here -- ``TransportAction`` issues
          ``MoveTorsoAction(TorsoState.HIGH)`` itself, and it is a sim-side defect.
        - **A single exception ends the run.** ``RobotDemonstration.run`` has no retry
          anywhere, while :mod:`demos.garmi_demo` catches ``GiskardException`` and
          retries. Against physics a transient infeasibility on the first motion is
          ordinary.

        :param context: The context the plan is about to be built against; its
            ``world`` is the one the controller is serving on a real run.
        """
        if self.execution_type is ExecutionType.SIMULATED:
            return
        self.sync_to_isaac(context)

    @staticmethod
    def sync_to_isaac(context: Context) -> None:
        """Put the objects upstream spawned into the twin into Isaac as well.

        ``populate_scene`` is upstream's and spawns ``bowl.stl`` and ``spoon.stl``
        into the *twin*, which is all a kinematic run needs. A run against physics
        needs them in the render too, or the hand closes on nothing and
        ``PickUpAction`` attaches the body anyway, because a kinematic attach cannot
        fail.

        The same two files, at the poses the twin already put them at -- so the two
        sides describe one object rather than two descriptions someone has to keep
        in step. That is the whole reason this replaced a hardcoded prop table: the
        table had to state the poses a second time, and the twin's and the render's
        drifted apart every time either moved.

        ``track=True``: these are objects physics decides about. Once released they
        fall, settle, and get knocked around, and the twin should follow rather than
        keep asserting where the plan last put them.

        Tracking only makes the sim *publish*, though; ``pull`` is what writes those
        poses into the twin. This pulls once here, after letting the objects settle,
        so the plan starts from where they came to rest rather than from where they
        were released. A plan that wants to know again later calls ``pull`` again --
        it is deliberately not automatic, because a pose applied mid-motion would
        move an object under the plan reaching for it.

        The colliders are per object and not interchangeable: ``convexDecomposition``
        holds the bowl's cavity open where a single hull would fill it in.
        """
        world = context.world
        sync = SceneSyncClient(context.ros_node)
        for name, stl, collider, mass in (
            (upstream.BOWL_NAME, upstream.BOWL_STL, "convexDecomposition", 0.058),
            (upstream.SPOON_NAME, upstream.SPOON_STL, "convexDecomposition", 0.05),
        ):
            body = world.get_body_by_name(name)
            pose = np.asarray(body.global_pose.to_np())
            sync.place(
                name,
                pose[:3, 3].ravel(),
                np.asarray(
                    HomogeneousTransformationMatrix(data=pose).to_quaternion().to_np()
                ).ravel(),
                mesh=stl,
                collider=collider,
                mass=mass,
                track=True,
            )
        print(f"  synced to Isaac: {sync.apply()}", flush=True)

        # Settle, then take the settled poses back. bowl.stl is released above the
        # worktop and spoon.stl inside a drawer; both drop a little before they rest.
        deadline = time.monotonic() + SETTLE_SECONDS
        while time.monotonic() < deadline:
            time.sleep(0.1)
        moved = sync.pull(world)
        print(
            "  pulled back: "
            + (", ".join(f"{n} {d * 1000:.1f} mm" for n, d in moved.items()) or "nothing"),
            flush=True,
        )


def main(execution_type: ExecutionType = ExecutionType.REAL) -> None:
    """Run the demonstration.

    :param execution_type: ``REAL`` drives the running giskard server and the Isaac
        scene behind it; ``SIMULATED`` is upstream's own kinematic path.
    """
    GarmiApartmentOnIsaac(
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
    main(ExecutionType.SIMULATED if arguments.simulated else ExecutionType.REAL)
