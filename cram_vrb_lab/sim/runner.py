"""The Isaac-side program every demo runs: build the setup, then step it forever.

``demos/sim.py`` is a five-line entry script over this module -- everything that
differs between robots and scenes is looked up in :mod:`cram_vrb_lab.setups`
rather than written out per combination.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app` has
   run: :func:`build` reaches into modules that import ``isaacsim.core`` at module
   scope. This module itself does not, so the entry script can import it early.
"""

import rclpy

from cram_vrb_lab.setups import get_setup, spawn_pose_from_args
from cram_vrb_lab.sim.isaac_app import READY_MARKER

SPINS_PER_STEP = 16
"""How many callbacks to drain per sim step.

``spin_once`` handles exactly ONE message per call, and with giskard streaming two
topics at ~20 Hz each, a single spin per sim tick falls behind and commands arrive
stale.
"""


def build(world, render, setup, spawn_pose, args):
    """Load the scene, spawn the robot at ``spawn_pose``, add the props, and
    return the ROS nodes.

    The order is the one every scene needs: scenery, robot, props, and only then
    the robot's park pose -- ``spawn_props`` calls ``world.reset()``, which throws
    away drive gains and poses set before it.
    """
    view = setup.viewport(spawn_pose) if setup.viewport else None
    setup.scene.load(
        world,
        render,
        camera_eye=view.eye if view else None,
        camera_target=view.target if view else None,
    )

    robot = setup.robot.spawn(world, render, spawn_pose)

    props = None
    if setup.wants_props(args.props):
        from cram_vrb_lab.scenes.props.isaac_props import spawn_props

        props = spawn_props(world, render, layout=setup.props.layout(spawn_pose))

    if setup.robot.park is not None:
        setup.robot.park(robot, world, render)

    if not rclpy.ok():
        rclpy.init(args=None)

    nodes = [setup.robot.ros_node(world, render, robot, args)]
    if props is not None:
        from cram_vrb_lab.scenes.props.isaac_props import PropsROS

        nodes.append(PropsROS(props))
    return nodes


def run(simulation_app, world, render, args):
    """Build the setup ``args`` selects and step it until the app is closed."""
    setup = get_setup(args.robot, args.scene)
    spawn_pose = spawn_pose_from_args(args)
    nodes = build(world, render, setup, spawn_pose, args)
    commanded = [node for node in nodes if node.receives_commands]

    # flush=True is load-bearing, not decoration. This is the marker
    # launcher.start_isaac_sim polls the log file for, and Isaac's own logging goes
    # to fd 1 from C++ while this print lands in Python's block buffer -- which the
    # loop below never writes enough to fill. Without the flush the sim comes up
    # fully, publishes every topic, and the notebook's first cell still sits there
    # until it times out.
    print(f"{setup.name} at {spawn_pose}: {READY_MARKER}", flush=True)

    try:
        while simulation_app.is_running():
            dt = world.get_rendering_dt()
            for node in nodes:
                node.apply_commands(dt)
            world.step(render=render)
            for _ in range(SPINS_PER_STEP):
                for node in commanded:
                    rclpy.spin_once(node, timeout_sec=0.0)
            for node in nodes:
                node.publish()
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()
