"""The Isaac-side program every demo runs: build the setup, then step it forever.

``demos/sim.py`` is a five-line entry script over this module -- everything that
differs between robots and scenes is looked up in :mod:`cram_vrb_lab.setups`
rather than written out per combination.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app` has
   run: :func:`build` reaches into modules that import ``isaacsim.core`` at module
   scope. This module itself does not, so the entry script can import it early.
"""

import time

import rclpy

from cram_vrb_lab.setups import get_setup, spawn_pose_from_args
from cram_vrb_lab.sim.isaac_app import READY_MARKER

SPINS_PER_STEP = 16
"""How many callbacks to drain per sim step.

``spin_once`` handles exactly ONE message per call, and with giskard streaming two
topics at ~20 Hz each, a single spin per sim tick falls behind and commands arrive
stale.
"""

RATE_REPORT_PERIOD = 5.0
"""Seconds between the loop-rate reports :func:`report_rate` prints."""

SLOW_RTF = 0.8
"""Below this real-time factor the report is raised to a warning.

Not a cosmetic threshold. The control cycle below feeds giskard, whose QP runs at
a fixed 15 Hz on the wall clock and never checks whether the feedback it reads is
new (its joint-state sync silently re-applies the last message). A cycle rate
below giskard's is a real-time controller closing its loop on stale state, which
it answers with overshoot -- visible as a robot that shakes instead of tracking.
0.8 leaves the nominal 25 Hz cycle comfortably above giskard's 15 Hz while
flagging anything that is starting to slip.
"""

SMOOTHING = 0.2
"""Weight of the newest sample in the smoothed per-cycle cost estimates.

The costs are spiky (an asset streaming in, a shader compiling); the report
should show the trend rather than one slow frame.
"""


def smooth(estimate, sample):
    """Exponentially smoothed cost estimate; the first sample seeds it."""
    return sample if estimate == 0.0 else estimate + SMOOTHING * (sample - estimate)


def probe_costs(world, render, frames=3):
    """Time physics and a frame separately, once, and print the split.

    :func:`report_rate` can only say that a cycle is slow; this says *what* is
    slow, which is the difference between a display problem (watch the livestream
    instead of a native window, shrink ``ISAAC_WINDOW``) and a physics one (fewer
    bodies, a larger ``physics_dt``) -- and the two want opposite fixes.

    Deliberately not merged into the loop: a fused ``world.step(render=True)``
    cannot be split after the fact, and hand-stepping physics for every cycle
    just to measure it costs more than it reports (see the note in :func:`run`).
    Here it is paid once, over a cycle's worth of steps, before the loop starts.

    Both numbers are bounds, in opposite directions, so read them as a ratio and
    let :func:`report_rate` say what a cycle really costs. Physics is an upper
    bound (hand-stepping pays a results fetch per step). The frame is a lower
    bound: a standalone ``render()`` does not do everything a fused update does,
    and it collapses to a few ms when nothing is consuming frames -- on an RTX
    3080 it read 5-6 ms whether the app was 1280x960 or 2560x1920, while the
    cycle those two actually cost was 18 ms and 33 ms.
    """
    physics_dt = world.get_physics_dt()
    substeps = max(round(world.get_rendering_dt() / physics_dt), 1)

    started = time.time()
    for _ in range(substeps):
        world.step(render=False)
    physics_seconds = time.time() - started

    frame_seconds = 0.0
    if render:
        started = time.time()
        for _ in range(frames):
            world.render()
        frame_seconds = (time.time() - started) / frames

    budget = substeps * physics_dt
    print(
        f"[sim] cost probe: physics {physics_seconds * 1e3:.0f} ms/cycle"
        f" ({substeps} x {physics_dt * 1e3:.0f} ms steps)"
        + (f", frame {frame_seconds * 1e3:.0f} ms" if render else ", not rendering")
        + f"  -- the cycle has {budget * 1e3:.0f} ms to run in real time."
        " Hand-stepping costs more than a fused step, so treat the physics"
        " number as an upper bound.",
        flush=True,
    )


def report_rate(fps, nominal_fps, work_seconds, render=True):
    """Print the control-cycle rate, the real-time factor, and the work per cycle.

    ``work_seconds`` is what ``world.step`` costs; the rest of the cycle is the
    ROS work and, when there is time left, the sleep that holds the loop to real
    time. Measured on an RTX 3080 with the GARMI apartment: 18 ms headless or
    livestreaming, 45 ms with a native Isaac window on the VNC desktop -- which
    is the whole difference between a controller that tracks and one that shakes.

    ``flush=True`` for the same reason as the ready marker below: Isaac writes to
    fd 1 from C++ while this goes through Python's block buffer, which the loop
    never writes enough to fill.
    """
    rtf = fps / nominal_fps if nominal_fps else 0.0
    slow = rtf < SLOW_RTF
    hint = (
        "  -- giskard closes its loop at 15 Hz on this feedback; a native window"
        " on a remote desktop is the usual cause"
        if render
        else "  -- giskard closes its loop at 15 Hz on this feedback; ISAAC_RENDER=0"
        " hand-steps physics and is slower here than rendering, not faster"
    )
    print(
        f"{'WARNING: ' if slow else ''}[sim] {fps:.1f} Hz  RTF {rtf:.2f}"
        f"  (work {work_seconds * 1e3:.0f} ms/cycle)" + (hint if slow else ""),
        flush=True,
    )


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
    probe_costs(world, render)
    print(f"{setup.name} at {spawn_pose}: {READY_MARKER}", flush=True)

    # One thread does everything, so a cycle of this loop is three things at once:
    # a step of physics, the point at which giskard's streamed commands are
    # consumed, and the point at which the joint states / odometry / TF giskard
    # closes its loop on are published. Its rate is therefore the control rate,
    # and it must stay above giskard's fixed 15 Hz (see SLOW_RTF).
    #
    # `world.step(render=True)` is one `app.update()`: the frame AND the physics
    # (rendering_dt / physics_dt substeps of it) in one blocking call. So the
    # controller runs at whatever rate the display manages, and a display that is
    # slow to present -- a native Isaac window on a VNC desktop, where every
    # frame is a CPU copy and a re-encode -- is paid for in control rate. On an
    # RTX 3080 with the GARMI apartment that is 18 ms per cycle headless (or
    # livestreaming, which renders offscreen and encodes on the GPU) against
    # 45 ms for the native window: 25 Hz against 17 Hz, either side of giskard's
    # 15 Hz. Hence the report, and hence the advice in demos/README.md to watch
    # the livestream on that desktop rather than a native window.
    #
    # Stepping physics separately (`world.step(render=False)` x substeps) to draw
    # less often was tried and measured: it costs MORE than it saves, because
    # each hand-driven step pays its own PhysX results fetch, which the substeps
    # inside one app.update() share -- 44 ms per cycle against those 18. Do not
    # reach for it again without measuring first.
    #
    # ISAAC_RENDER=0 has no fused call available and must hand-step, which is why
    # it is a fallback for hardware that cannot render at all and NOT a way to go
    # faster: it pays that same penalty, and it is the one path where a cycle
    # costs more without a frame in it than the other paths cost with one.
    physics_dt = world.get_physics_dt()
    rendering_dt = world.get_rendering_dt()
    steps_per_cycle = 1 if render else max(round(rendering_dt / physics_dt), 1)
    cycle_dt = rendering_dt if render else steps_per_cycle * physics_dt
    work_seconds = 0.0
    cycles = 0
    window_start = deadline = time.time()
    try:
        while simulation_app.is_running():
            for node in nodes:
                node.apply_commands(cycle_dt)

            started = time.time()
            # One fused call when rendering (physics substeps included), else a
            # cycle's worth of physics steps by hand -- either way the cycle
            # advances the cycle_dt that apply_commands was just told about.
            for _ in range(steps_per_cycle):
                world.step(render=render)
            work_seconds = smooth(work_seconds, time.time() - started)

            for _ in range(SPINS_PER_STEP):
                for node in commanded:
                    rclpy.spin_once(node, timeout_sec=0.0)
            for node in nodes:
                node.publish()

            # Hold the cycle to real time. Nothing else does, and a machine that
            # can step this scene faster than real time -- which is every machine
            # that is not rendering, the sim spends 18 of its 40 ms on physics
            # here -- would otherwise advance sim time faster than the wall clock
            # giskard plans in, and the base, which integrate_base dead-reckons
            # over the sim step, would overshoot every goal by that factor. A
            # cycle that is already late does not try to catch up: the deadline
            # is pushed to now, so a hitch costs one slow cycle rather than a
            # burst of fast ones.
            deadline += cycle_dt
            now = time.time()
            if now < deadline:
                time.sleep(deadline - now)
            else:
                deadline = now

            cycles += 1
            elapsed = time.time() - window_start
            if elapsed >= RATE_REPORT_PERIOD:
                report_rate(cycles / elapsed, 1.0 / cycle_dt, work_seconds, render)
                cycles, window_start = 0, time.time()
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()
