#!/usr/bin/env python
"""Isaac Sim side of every demo: a robot in a scene, controlled over ROS 2.

One entry script for all combinations -- ``--robot`` and ``--scene`` pick one out
of :mod:`cram_vrb_lab.setups`, which is also where a new combination is added.
The scene publishes the interface a real robot would (joint states, odometry, TF,
camera) and subscribes to the commands giskard streams, so the giskard server and
the notebooks work the same way whichever setup is running:

- publishes ``<robot>/joint_states``, and for a mobile base ``/odom`` and TF
  (odom->base_link->links, plus the fixed camera-frame chain as static tf)
- publishes the head camera per ``--camera``: ``/head_camera/image_raw`` (rgb8)
  and/or ``/head_camera/depth/image_raw`` (32FC1, metres) with camera_info,
  stamped in ``camera_color_optical_frame``
- subscribes ``<robot>/joint_velocity_cmd`` (giskard's streamed velocities,
  integrated into position targets each sim step), ``<robot>/gripper_command``
  (Float64), and for a mobile base ``<robot>/cmd_vel`` (Twist, kinematic base
  with a 1 s watchdog)
- with props: spawns the graspable pick-and-place cube and publishes its
  ground-truth pose on ``/props/pick_cube_pose``

Run with the Isaac Sim python (or from the demo notebooks):
    binder/isaacsim_python_wrapper.sh demos/sim.py \
        [--robot NAME] [--scene NAME] [--camera MODE] [--props]
where MODE is rgb (default), depth, both, or none.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cram_vrb_lab.sim.isaac_app import (
    create_simulation_app,
    livestream_enabled,
    parse_scene_args,
    physics_engine,
    render_enabled,
    use_newton,
)

ARGS = parse_scene_args()
simulation_app = create_simulation_app()  # must run before any isaacsim.core import
RENDER = render_enabled()

from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")

# ISAAC_PHYSICS=newton runs the scene on Newton instead of PhysX. The extension
# carries `auto_switch_on_startup = true`, so loading it is the switch; the
# explicit call is what makes a failed switch loud instead of a scene that
# quietly went on running on PhysX.
if use_newton():
    from isaacsim.core.simulation_manager import SimulationManager

    enable_extension("isaacsim.physics.newton")
    # The tensor views (Articulation, and everything the robot bridges read the
    # sim through) come from a second extension: without it the engine switches
    # and then the first world.reset() dies on "Failed to find simulation
    # backend 'newton'". The newton experience file loads both.
    enable_extension("isaacsim.physics.newton.tensors")
    if not SimulationManager.switch_physics_engine("newton"):
        raise RuntimeError("could not switch the physics engine to Newton")
    # Newton is a GPU engine: on the CPU it runs this scene at a fifth of real
    # time. A GPU pipeline forces Isaac's array backend from numpy to torch, so
    # the numpy this repo speaks is bridged in numpy_bridge (installed after the
    # first reset, which is where the backend actually flips).
    # ISAAC_PHYSICS_DEVICE=cpu keeps the slow, unbridged path.
    SimulationManager.set_device(os.environ.get("ISAAC_PHYSICS_DEVICE", "cuda:0"))

    # MuJoCo-Warp sizes its contact and constraint arrays up front, and the
    # defaults (200 contacts, 1200 constraints per world) are for a robot on a
    # ground plane, not a robot in a furnished flat: this scene peaks above 3000
    # contacts and spends the run printing "Number of Newton contacts exceeded
    # MJWarp limit" while dropping the rest, which is both wrong and slow.
    # A whole MuJoCoSolverConfig, not two fields poked into the config that is
    # already there: Newton replaces a solver config that is not of the solver's
    # own type with a default-constructed one when it initialises, and the two
    # numbers would go with it.
    from isaacsim.physics.newton import (
        MuJoCoSolverConfig,
        configure_newton,
        get_newton_config,
    )

    newton_config = get_newton_config()
    newton_config.solver_cfg = MuJoCoSolverConfig(
        nconmax=int(os.environ.get("NEWTON_NCONMAX", 8192)),
        njmax=int(os.environ.get("NEWTON_NJMAX", 8192)),
    )
    configure_newton(newton_config)
    print(f"[sim] physics engine: {SimulationManager.get_active_physics_engine()} "
          f"(nconmax {newton_config.solver_cfg.nconmax}, "
          f"njmax {newton_config.solver_cfg.njmax})")
# Shift + left-drag in the viewport to grab and push rigid bodies while the sim
# runs. The GUI app (`/isaac-sim/isaac-sim.sh`) has it because its experience file
# pulls in omni.physx.bundle; the experience a SimulationApp loads
# (isaacsim.exp.base.python.kit) does not, so it has to be asked for here. Only
# worth loading when there is a viewport to interact with.
if RENDER:
    enable_extension("omni.physx.ui")

# Watch the viewport from the WebRTC client instead of a local window
# (ISAAC_LIVESTREAM=1). The extension is `omni.kit.livestream.app`, which streams
# the whole application framebuffer over WebRTC (signalling on 49100, stream on
# 47998) -- the same one `apps/isaacsim.exp.full.streaming.kit` pulls in, and what
# `isaacsim-webrtc-streaming-client` connects to. Isaac Sim 6.1 dropped
# `omni.services.livestream.nvcf`, the NVCF-deployment extension this used to
# enable, and an extension that is not there fails the whole dependency
# resolution rather than the stream alone:
#
#   [Error] [omni.ext.plugin] Failed to resolve extension dependencies.
#     * No versions of omni.services.livestream.nvcf that satisfies: ...
if livestream_enabled():
    simulation_app.set_setting("/app/window/drawMouse", True)
    enable_extension("omni.kit.livestream.app")

my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=8 / 200)
my_world.reset()

if use_newton():
    from cram_vrb_lab.sim.numpy_bridge import install as install_numpy_bridge

    if install_numpy_bridge():
        print("[sim] numpy -> torch bridge installed for the GPU backend")

from cram_vrb_lab.sim.runner import run

run(simulation_app, my_world, RENDER, ARGS)
