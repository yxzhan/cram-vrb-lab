#!/usr/bin/env python
# coding: utf-8

# # giskard × Isaac Sim: Stretch control demo
# 
# Drive the Stretch robot in the apartment scene through
# [giskardpy](https://github.com/cram2/cognitive_robot_abstract_machine)'s
# closed-loop whole-body QP controller: joint-space, Cartesian end-effector,
# base, and gripper goals.
# 
# **Kernel**: select **CRAM** (registered by the Docker image; register manually by
# copying `binder/cram_python_wrapper.sh` to `~/.local/bin/` and
# `binder/cram-kernel.json` to `~/.local/share/jupyter/kernels/cram/kernel.json`).
# 
# The two cells below start the simulation and the giskard server as background
# processes. Skip them if you already started these in a terminal.
# 

# ## Start the Isaac Sim simulation
# 
# First startup can take a few minutes (shader compilation). Progress is written
# to `/tmp/apartment_sim.log`.
# 

# In[ ]:


import subprocess
import sys
import time
from pathlib import Path

REPO = Path.cwd().resolve().parent  # this notebook lives in giskard_stretch/


def start(args, log_path, marker, timeout):
    """Start a background process and wait until `marker` appears in its log."""
    proc = subprocess.Popen(args, stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if marker in Path(log_path).read_text(errors='ignore'):
            print(f'ready after {time.time() - t0:.0f}s')
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f'process exited early -- check {log_path}')
        time.sleep(2)
    raise TimeoutError(f'{marker!r} not found in {log_path} -- check that log')


sim_proc = start([f'{REPO}/binder/isaacsim_python_wrapper.sh',
                  f'{REPO}/giskard_stretch/apartment.py'],
                 '/tmp/apartment_sim.log', 'StretchROS node ready.', timeout=900)


# ## Start the giskard server
# 
# Runs on this kernel's interpreter (`sys.executable` = the CRAM venv);
# logs to `/tmp/giskard_server.log`.
# 

# In[ ]:


giskard_proc = start([sys.executable, f'{REPO}/giskard_stretch/giskard_stretch_isaac.py'],
                     '/tmp/giskard_server.log', 'giskard is ready', timeout=300)


# ## Connect to giskard
# 
# Fetches the world model from the server and defines a small execution helper.
# `add_end_conditions` holds a reached goal for one extra second before ending the
# motion (settle phase): giskard's behavior tree keeps publishing the last commanded
# velocities for about a second between goal completion and its terminate-zero
# message, so ending immediately lets the base coast past the goal.
# 

# In[ ]:


import sys
sys.path.insert(0, str(REPO / 'giskard_stretch'))

from giskard_client import add_end_conditions, connect
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.motion_statechart import MotionStatechart

giskard = connect('giskard_notebook_client')
world = giskard.world
print('connected, robot:', giskard.robot_name)


def run_goal(task, timeout=60.0):
    """Execute a single motion task with settle phase + timeout."""
    msc = MotionStatechart()
    msc.add_node(task)
    add_end_conditions(msc, task, timeout_seconds=timeout)
    giskard.execute(msc)
    reached = msc.observation_state[task] == ObservationStateValues.TRUE
    print('goal reached:', reached)
    return reached


# ## 1. Joint-space goal
# 
# Command joint target positions directly. Available joints: see
# `stretch_joints.CONTROLLED_JOINTS`. Threshold 0.02: this pipeline settles within
# 0.01-0.02 of the goal, so the default 0.01 latches unreliably.
# 

# In[ ]:


from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.joint_state import JointState

goal = {
    world.get_connection_by_name('joint_lift'): 0.8,
    world.get_connection_by_name('joint_wrist_yaw'): 0.0,
}
run_goal(JointPositionList(goal_state=JointState.from_mapping(goal), threshold=0.02),
         timeout=30.0)


# ## 2. Gripper open / close
# 

# In[ ]:


FINGER_OPEN, FINGER_CLOSED = 0.109, 0.0


def gripper(position):
    goal = {
        world.get_connection_by_name('joint_gripper_finger_left'): position,
        world.get_connection_by_name('joint_gripper_finger_right'): position,
    }
    return run_goal(JointPositionList(goal_state=JointState.from_mapping(goal),
                                      threshold=0.02), timeout=30.0)


gripper(FINGER_OPEN)


# In[ ]:


gripper(FINGER_CLOSED)


# ## 3. Cartesian end-effector goal (arm only)
# 
# Translate `link_grasp_center` in its own frame, keeping the current orientation.
# `root_link=base_link` restricts the motion to the arm (base stays put).
# Note: self-collision avoidance is not enabled -- avoid goals that press the
# gripper into the robot's own body.
# 

# In[ ]:


from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.spatial_types.spatial_types import Pose, Vector3

tip = world.get_kinematic_structure_entity_by_name('link_grasp_center')
base = world.get_kinematic_structure_entity_by_name('base_link')

goal_pose = Pose(position=Vector3(0.0, 0.0, 0.15), reference_frame=tip)  # 15 cm up
run_goal(CartesianPose(root_link=base, tip_link=tip, goal_pose=goal_pose, threshold=0.03))


# ## 4. Cartesian end-effector goal (whole body)
# 
# With `root_link=world.root` the base joins the motion -- if the target is out of
# the arm's reach, the base drives to make up the difference.
# 

# In[ ]:


goal_pose = Pose(position=Vector3(0.4, 0.0, 0.0), reference_frame=tip)
run_goal(CartesianPose(root_link=world.root, tip_link=tip, goal_pose=goal_pose,
                       threshold=0.03), timeout=90.0)


# ## 5. Base goal
# 
# `DifferentialDriveBaseGoal` (orient -> drive -> orient) is the right idiom for a
# non-holonomic base; a plain `CartesianPose` on `base_link` oscillates around the
# goal. The 5 cm threshold is the usual tolerance for a mobile base.
# 

# In[ ]:


from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal

goal_pose = Pose(position=Vector3(0.5, 0.0, 0.0), reference_frame=base)  # 0.5 m ahead
run_goal(DifferentialDriveBaseGoal(goal_pose=goal_pose, threshold=0.05))


# ## Shutdown
# 
# Stop the giskard server and the simulation (only if they were started from this
# notebook).
# 

# In[ ]:


for proc in (giskard_proc, sim_proc):
    proc.terminate()
    proc.wait(timeout=30)
print('stopped')


# ## Troubleshooting
# 
# - **Gripper stuck after touching an object**: reset it through the sim's native
#   interface: `ros2 topic pub --once /stretch/gripper_command std_msgs/msg/Float64 "{data: 0.05}"`
# - **Arm in a twisted pose**: send a plain `/stretch/joint_command` (JointState) to
#   move it back to a neutral configuration
# - giskard is a whole-body controller: even an arm-only goal may slightly adjust
#   other controlled joints
# - More details in `giskard_stretch/README.md`
# 
