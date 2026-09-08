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

Needs the Isaac scene and the giskard server already up, and the scene started with
``ISAAC_TRANSPORT_PROPS=1`` so the two objects the plan carries are actually there::

    ISAAC_TRANSPORT_PROPS=1 binder/isaacsim_python_wrapper.sh demos/sim.py --robot garmi --scene garmi_apartment --spawn-position 0.0 5.0 0.0259 --spawn-yaw 1.5707963267948966
    binder/cram_python_wrapper.sh demos/giskard_server.py --robot garmi --scene garmi_apartment --control-hz 15 --spawn-position 0.0 5.0 0.0259 --spawn-yaw 1.5707963267948966
    binder/cram_python_wrapper.sh demos/garmi_transport_demo.py

Without that variable Isaac renders the flat with nothing in it, the twin still gets
its bowl and spoon from ``populate_scene``, and the run closes the hand on air --
which is exactly the phantom-object run
:data:`~cram_vrb_lab.scenes.garmi_apartment.constants.TRANSPORT_PROPS` exists to
prevent. Both sides read that one table, so the twin's belief and the physics agree
by construction; :meth:`GarmiApartmentOnIsaac.align_with_isaac` is what applies it to
the twin.

``--simulated`` runs upstream's own kinematic path unchanged and needs neither
process, which is the reference to compare a real run against.
"""


from __future__ import annotations

import argparse
import importlib.util
import sys
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

from cram_vrb_lab.scenes.garmi_apartment.constants import TRANSPORT_PROPS

# Imported for its side effect. Upstream's build_context passes
# AlternativeMotion.discover_all(), which returns every AlternativeMotion subclass that
# has been *loaded*, so importing this module is what puts GARMI's three motions --
# the gripper and opening/closing a container, each carrying the CountSeconds deadline
# giskard has no equivalent of -- into upstream's context without overriding it.
import cram_vrb_lab.robots.garmi.motions  # noqa: F401

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
        rather than a fork of the plan. So far there is one: :meth:`align_with_isaac`.

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
        if self.execution_type is not ExecutionType.SIMULATED:
            self.align_with_isaac(context)

    @staticmethod
    def align_with_isaac(context: Context) -> None:
        """Move the twin's bowl and spoon to where Isaac actually simulates them.

        ``populate_scene`` puts them at upstream's poses, which were written against a
        world that is stepped kinematically and contains nothing else. Two of those
        numbers do not survive contact with this scene:

        - ``BOWL_POSE``'s z of 1.0 is not this worktop. It raycasts at 0.945, and the
          bowl mesh carries its origin at its bounding-box centre, so the twin
          believes the bowl floats 2 cm above a surface Isaac has it resting on. A
          reach aimed at the belief closes the hand above the rim.
        - ``BOWL_POSE``'s x/y stand where nothing is. Isaac puts the bowl where
          :data:`~cram_vrb_lab.scenes.garmi_apartment.constants.TRANSPORT_PROPS` says,
          in the free band of worktop between the kitchen props' two rows.

        The spoon moves too, and by less: upstream's drawer-relative pose resolves to
        within 16 mm of where
        :data:`~cram_vrb_lab.scenes.garmi_apartment.constants.TRANSPORT_PROPS` puts
        it. Applied anyway rather than skipped -- "close enough" is a claim that
        should be re-derived every run, not assumed.

        The connection is left alone, so the spoon keeps hanging off ``drawer_1`` and
        still travels with it when the plan pulls it open; only the pose it hangs at
        changes. ``Connection6DoF.origin`` takes a transform in any frame and converts
        into the parent's, which is what lets one ``map`` constant drive both.

        Only for a run that is driving Isaac. ``--simulated`` builds its own world
        with no physics behind it, and there upstream's poses are the correct ones --
        nothing is standing anywhere for them to disagree with.
        """
        world = context.world
        for prop in TRANSPORT_PROPS:
            body = world.get_body_by_name(prop.name)
            box = body.collision.as_bounding_box_collection_in_frame(
                body
            ).bounding_box()

            # position[2] is the *surface*; the body's origin is its box centre, so
            # it sits half a box above it -- the same grounding the Isaac side does by
            # measuring, in closed form here because the twin has no physics to settle
            # it. In map for both, ``inside`` included: see TransportProp.position for
            # the 208 mm this cost when it was stated in the drawer's frame.
            target = HomogeneousTransformationMatrix.from_xyz_rpy(
                prop.position[0],
                prop.position[1],
                prop.position[2] + box.dimensions[2] / 2,
                yaw=prop.yaw,
                reference_frame=world.root,
            )

            before = np.asarray(body.global_pose.to_np())[:3, 3].ravel()
            body.parent_connection.origin = target
            after = np.asarray(body.global_pose.to_np())[:3, 3].ravel()
            print(
                f"  {prop.name}: {np.round(before, 4)} -> {np.round(after, 4)}"
                f"  (moved {np.linalg.norm(after - before) * 1000:.1f} mm)",
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
