"""The scene's own articulation -- drawers, cabinet doors, room doors -- measured in
Isaac and fed to giskard the way the robot's joints are.

Without this the two sides each keep their own copy of every container: giskard's
``Open`` drives the drawer's joint to its limit in its own model, the twin hears that
over ``/world_sync``, and nothing ever asks Isaac, where the handle may have slipped
out of the gripper with the drawer half open. With it, Isaac is the only writer of
these degrees of freedom, exactly as it is of the robot's: the sim publishes
:data:`SCENE_JOINT_STATES_TOPIC` every cycle
(:class:`~cram_vrb_lab.sim.scene_joints_bridge.SceneJointsROS`), giskard overwrites
its state with it every control cycle and every idle cycle
(:func:`cram_vrb_lab.control.giskard_server.with_scene_joint_states`), and the twin
follows giskard over ``/world_sync`` as it already does for the robot.

Plain data only: both the Isaac python and the CRAM venv import this module.
"""

from __future__ import annotations

from dataclasses import dataclass

SCENE_JOINT_STATES_TOPIC = "/scene/joint_states"
"""``sensor_msgs/JointState``, named by the twin's connection names, in the twin's
units (m, rad) and sign."""


@dataclass(frozen=True)
class SceneJoint:
    """One scene joint, as each side names and measures it."""

    usd_joint: str
    """The ``PhysicsRevoluteJoint`` / ``PhysicsPrismaticJoint`` prim, relative to the
    scene's root prim."""

    connection: str
    """The twin's connection name, i.e. what giskard's world calls the joint."""

    sign: float = 1.0
    """Maps the USD joint coordinate onto the twin's. Not always +1: the two
    descriptions pick their own joint axes, and a drawer the asset slides along -Y
    (``[-0.466, 0]``) the MJCF slides along +X (``[0, 0.466]``). Read off the limits,
    which both sides give with 0 = shut."""
