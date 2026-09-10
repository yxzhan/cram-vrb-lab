"""Put the scene back where it started, without restarting the sim.

Repeating a task means repeating its *initial conditions*, and the only way to get
them back used to be restarting Isaac -- two to three minutes of asset loading for
a run that takes twenty seconds. This is the same thing in about a second:

    from cram_vrb_lab.sim.scene_reset import SceneResetClient

    reset = SceneResetClient(node)
    print(reset())          # blocks until the sim says it is back

**Not** ``world.reset()``. That restores the state physics *started* from, which is
the wrong state twice over: the props were spawned hovering and only settled after
120 steps, so they would be dropped again; and it re-authors the drive parameters
off the prims, throwing away the gains
:func:`~cram_vrb_lab.robots.garmi.isaac_node.move_to_park` sets (its own docstring
says so). What this restores instead is a *snapshot* taken once the scene was fully
built, settled and parked -- the state a demo actually begins from.

The twin half is :func:`reset_context`, and it is deliberately a separate call: it
touches nothing Isaac owns, and a plan may well want one without the other. So is
:func:`cancel_motion`, which stops the goal giskard may still be executing -- run it
first, or the old goal drives out of the pose the reset just restored.
"""

from __future__ import annotations

from std_srvs.srv import Trigger

RESET_SERVICE = "/cram_vrb_lab/reset_scene"
"""Where :class:`SceneResetROS` listens.

A ``std_srvs/Trigger`` because a reset carries no arguments -- Isaac already knows
what the scene looked like, so nothing has to cross the boundary but the word "go".
That also means no interface package to build into ``ros2_ws``, which is what made
:mod:`cram_vrb_lab.sim.scene_sync` settle for a topic and an ack instead.
"""

DEFAULT_TIMEOUT = 30.0
"""Seconds to wait for the reset to come back.

Longer than the work needs -- restoring a snapshot is a few dozen prim writes -- but
the service is answered from the sim loop, so the wait also covers a sim that is
mid-step on a slow frame.
"""


class SceneResetClient:
    """Asks the sim to restore its snapshot, and waits for it.

    :param node: an rclpy node that is already being spun; the response arrives on
        that node's executor, so a client on an unspun node will always time out.
    """

    def __init__(self, node, timeout: float = DEFAULT_TIMEOUT):
        self.node = node
        self.timeout = timeout
        self._client = node.create_client(Trigger, RESET_SERVICE)

    def __call__(self, timeout: float | None = None) -> str:
        """Reset the sim; returns what it reported.

        :raises TimeoutError: if the service never appears, or never answers. Both
            mean the sim is not running this bridge rather than that the reset
            half-happened: the sim restores a snapshot in one piece, between two
            physics steps.
        :raises RuntimeError: if the sim answered but reported failure.
        """
        wait = self.timeout if timeout is None else timeout
        if not self._client.wait_for_service(timeout_sec=wait):
            raise TimeoutError(
                f"no {RESET_SERVICE} service -- is the sim running with the "
                "SceneResetROS bridge?"
            )
        future = self._client.call_async(Trigger.Request())
        deadline = self.node.get_clock().now().nanoseconds + int(wait * 1e9)
        while not future.done():
            if self.node.get_clock().now().nanoseconds > deadline:
                raise TimeoutError(f"{RESET_SERVICE} did not answer in {wait:g}s")
        response = future.result()
        if not response.success:
            raise RuntimeError(f"scene reset failed: {response.message}")
        return response.message


def cancel_motion(context, timeout: float = 5.0) -> str:
    """Stop the goal giskard is still executing; returns what was found.

    The third half of a reset, and the one neither the sim nor the twin covers. A goal
    lives on the **giskard server**: interrupting the cell that waits for it only drops
    the wait, and restarting the notebook does not touch it either. The server goes on
    running the motion it was handed -- so a scene reset lands in the middle of a plan
    that is still driving, and the robot walks straight back out of the pose the reset
    just restored, continuing the operation that was interrupted.

    Two leftovers, and the second is why this is not one line:

    - **The goal.** ``cancel_goal_async`` asks the server to stop.
      ``ActionServerHandler.cancel_callback`` accepts every cancel and ``ControlLoop``
      answers ``raise_if_canceled`` by stopping, which zeroes the commanded velocities,
      so the robot halts rather than coasting on a latched twist.
    - **The result nobody read.** A goal that ends -- cancelled, aborted, or simply
      finished after its client walked away -- still delivers its result into
      ``MyActionClient.result``, where it stays because the ``get_result`` that would
      have consumed it is gone. The *next* goal then awaits ``self.result`` and finds
      that one already there, so the next plan fails instantly with the previous run's
      ``ExecutionCanceledException``. The mailbox is emptied here as well.

    Call it **before** the sim reset: cancelling after it has already let the old goal
    command a step or two into the restored scene.

    :param context: the CRAM context whose ``giskard_wrapper`` sent the goal. Note that
        reading that property builds a wrapper when the context has never run a motion,
        which waits for giskard's action server.
    :param timeout: seconds to wait for the cancel to be answered and the goal to end.
    """
    import time

    from giskardpy.middleware.ros2.exceptions import NoActiveGoalToCancelError

    client = context.giskard_wrapper._client
    goal_id = client._current_goal_id
    deadline = time.monotonic() + timeout
    try:
        future = context.giskard_wrapper.cancel_goal_async()
    except NoActiveGoalToCancelError:
        report = "no active goal"
    else:
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            report = f"goal #{goal_id}: cancel request went unanswered"
        else:
            # The server answers the cancel before the motion is actually over; the goal
            # has ended when its own done callback drops the handle.
            while client._goal_handle is not None and time.monotonic() < deadline:
                time.sleep(0.02)
            report = (
                f"cancelled goal #{goal_id}"
                if client._goal_handle is None
                else f"goal #{goal_id}: cancelled, but it has not ended yet"
            )
    if client.result is not None:
        client.result = None
        report += ", dropped a stale result"
    return report


def reset_context(world, detach_to_root: bool = True, close_containers: bool = True) -> dict:
    """Clear what a finished plan left in the twin; returns what went.

    The twin half of a reset, and the half that is easy to forget because nothing
    about it is visible in the render. A world here belongs to the **giskard
    server**, which outlives the demo process, so none of this is cleaned up by
    restarting the notebook -- a second run starts on top of the first one's
    leftovers.

    Three kinds of leftover:

    - **Perceived bodies and their annotations**, from
      :func:`~cram_vrb_lab.perception.twin_objects.add_detections`. Handled by
      ``clear_detections``, which drops both -- see there for why an annotation
      outliving its body is a crash rather than clutter.
    - **Objects still parented to a gripper.** ``PickUpAction`` re-parents what it
      grasps onto the tool frame and only ``PlaceAction`` puts it back, so a plan
      that failed between the two leaves the object riding the hand for every run
      that follows.
    - **Drawers and doors the twin still believes are open.** This is the one that
      a sim reset cannot help with, because the two sides are not connected here:
      ``GarmiROS.publish_joint_states`` sends the *robot articulation's* degrees of
      freedom and nothing else, so the apartment's containers have no path from
      Isaac back to the twin at all. Reset the scene and the drawer shuts in the
      render while giskard goes on planning around one that is 0.466 m out.

    :param detach_to_root: whether to re-parent anything hanging off a robot link
        back to the world root. Off if a caller wants to inspect what was carried.
    :param close_containers: whether to drive every non-robot articulated joint
        back to 0. Sound for this apartment rather than assumed: all eleven of its
        containers put 0 at an *end* of their range -- drawers ``[0, 0.466]``,
        cabinet doors ``[0, 90]``, ``door_0_leaf`` ``[-90, 0]`` -- so zero is
        unambiguously "shut" for each. A scene whose joints do not is why this can
        be turned off.
    """
    from semantic_digital_twin.robots.robot_parts import AbstractRobot

    from cram_vrb_lab.perception.twin_objects import clear_detections

    report = {"detections": clear_detections(world), "detached": [], "closed": []}
    # Robots specifically, not "everything with bodies". ``Drawer`` and ``Handle``
    # have a ``bodies`` too, so a set built from every annotation swallows the very
    # containers this is meant to close -- and only once a demo has annotated them,
    # which is to say only from the second run onwards, exactly when a reset is
    # what you reached for.
    robot_bodies = {
        id(body)
        for robot in world.get_semantic_annotations_by_type(AbstractRobot)
        for body in robot.bodies
    }

    if close_containers:
        for connection in world.connections:
            if getattr(connection, "raw_dof", None) is None:
                continue
            # The robot's own joints are streamed back from the sim every step, so
            # writing them here would be overwritten within a cycle anyway -- and
            # would fight the controller in the meantime. Everything else in the
            # apartment has no such path and is this function's job.
            if any(
                id(body) in robot_bodies
                for body in (connection.parent, connection.child)
            ):
                continue
            if connection.position == 0.0:
                continue
            report["closed"].append(
                f"{connection.name.name}={connection.position:.3f}"
            )
            connection.position = 0.0
            connection.velocity = 0.0
        if report["closed"]:
            # The position setter writes the degree of freedom and stops there, so
            # nothing downstream -- forward kinematics, or the WorldSynchronizer
            # that carries this to giskard -- hears about it without this.
            world.notify_state_change()

    if not detach_to_root:
        return report

    carried = [
        body
        for body in world.bodies
        if body.parent_kinematic_structure_entity is not None
        and id(body.parent_kinematic_structure_entity) in robot_bodies
        and id(body) not in robot_bodies
    ]
    for body in carried:
        world.move_branch(body, world.root)
        report["detached"].append(body.name.name)
    return report
