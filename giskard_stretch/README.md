# giskardpy × Isaac Sim Stretch 集成

用 [giskardpy](../cognitive_robot_abstract_machine/giskardpy)（闭环 QP 全身控制）控制
`examples/apartment.py` 仿真里的 Stretch 机器人。纯增量集成：不修改 `apartment.py`、
giskardpy 核心或 semantic_digital_twin。

## 架构

```
Isaac Sim (examples/apartment.py)                      giskard 服务端 (本目录)
  /stretch/joint_states       ─────────────────────►  sync_joint_state_topic
  /odom                       ─────────────────────►  sync_odometry_topic (DiffDrive)
  /stretch/cmd_vel            ◄─────────────────────  底盘 Twist (闭环)
  /stretch/joint_velocity_cmd ◄─────────────────────  关节速度 (Float64MultiArray, ~20Hz)

  demo 客户端 ── JsonAction(/giskard/command) ──► giskard
```

- giskard 闭环输出**关节速度**，仿真在每个物理步里把它积分成位置目标
  （`StretchROS.integrate_joint_velocities`，用精确的仿真 dt；带反饱和钳制、
  限位钳制、0.5s 静默保持）。
- 底盘 Twist 直接对接 `/stretch/cmd_vel`；`integrate_base` 里有 1 秒的 cmd_vel
  看门狗——流式发布（giskard/Nav2）停止后底盘会自动停住，不再永久锁存。
  注意：这也意味着 notebook 里单次发布、靠锁存持续运动的手动 teleop 会在
  1 秒后停下（需要持续运动就持续发布）。
- URDF 用仿真自带的 `usd/stretch/stretch.urdf`，加载时做两处修补
  （`stretch_joints.load_patched_urdf`）：插入语义模型需要的
  `link_straight_gripper`；mesh 路径改为绝对路径（mesh 已从
  hello-robot/stretch_urdf 拷贝到 `usd/stretch/meshes/`）。
- 关节顺序契约在 `stretch_joints.CONTROLLED_JOINTS`（速度消息不带关节名，
  giskard 端和桥接端必须一致）。

## 一次性安装（已完成，记录备查）

```bash
# ROS 依赖
sudo apt-get install -y ros-jazzy-rclpy-message-converter ros-jazzy-py-trees-ros
# venv 依赖
/home/jovyan/isaacsim-giskard/cognitive_robot_abstract_machine/.venv/bin/pip install py_trees argcomplete
# giskard 的 action 消息包（本仓库自带定义）
source /opt/ros/jazzy/setup.bash
cd giskard_stretch/ros2_ws && colcon build --packages-select json_msgs
```

## 启动顺序

每个终端先 source：

```bash
source /opt/ros/jazzy/setup.bash
source /home/jovyan/isaacsim-giskard/giskard_stretch/ros2_ws/install/setup.bash
```

1. **Isaac Sim**：`~/.local/bin/isaacsim_python_wrapper.sh examples/apartment.py`
2. **giskard 服务端**（monorepo venv）：
   ```bash
   /home/jovyan/isaacsim-giskard/cognitive_robot_abstract_machine/.venv/bin/python \
       giskard_stretch/giskard_stretch_isaac.py
   ```
   等待日志出现 `giskard is ready`。

## Demo

都用 venv python 运行：

```bash
PY=/home/jovyan/isaacsim-giskard/cognitive_robot_abstract_machine/.venv/bin/python
$PY giskard_stretch/demo_joint_goal.py 0.8            # 升降到 0.8m
$PY giskard_stretch/demo_gripper.py open|close        # 夹爪开合
$PY giskard_stretch/demo_cartesian_goal.py 0 0 0.15   # 末端沿自身坐标系平移（仅手臂）
$PY giskard_stretch/demo_cartesian_goal.py --full-body 0.4 0 0   # 全身（底盘参与）
$PY giskard_stretch/demo_base_goal.py 0.5 0           # 底盘（DifferentialDriveBaseGoal）
```

自定义运动目标参照 `giskard_client.py` + 各 demo 的写法；任务类型见
`giskardpy/src/giskardpy/motion_statechart/tasks/`，文档示例
`giskardpy/doc/examples/`。

## 实测精度与已知问题

| 通路 | 实测 |
|---|---|
| 关节目标 | 收敛到 0.01–0.02（demo 阈值 0.02） |
| 末端笛卡尔 | 3cm/0.03rad 阈值下可靠到达 |
| 底盘 | ~5–8cm（阈值 5cm + 稳定相残差） |

- **稳定相（settle）仍然建议保留**：EndMotion 触发后 giskard 的行为树还会持续
  ~1 秒发布最后一帧速度才发终止零速。cmd_vel 看门狗（1s）能兜底，但
  `giskard_client.add_end_conditions` 让目标保持 1 秒（此时 QP 输出趋零）再结束，
  停得更准。自写脚本请沿用该助手。
- **没有启用自碰撞规避**（`WorldWithStretchConfigDiffDrive.setup_collision_config`
  为空）。不要把末端目标设到会撞自身/场景的位置（比如手腕弯曲时向下压）。
  夹爪与物体接触可能把手指"卡死"，此时用仿真原生接口复位：
  ```bash
  ros2 topic pub --once /stretch/gripper_command std_msgs/msg/Float64 "{data: 0.05}"
  ```
  必要时用 `/stretch/joint_command` 直接摆回标准姿态。
- giskard 是**全身控制器**：即使是"仅手臂"的目标，其他受控关节（含底盘）也可能
  被 QP 轻微调整。需要绝对不动底盘时，先不要给底盘发 cmd_vel 的能力
  （或接受几毫米级的挪动）。
- giskard 工作期间不要同时手动 teleop（`/stretch/cmd_vel`、`/stretch/ee_command`、
  `/stretch/joint_command` 会互相覆盖）。
- giskard 与仿真各自发布 TF（`odom→base_link` 有两个来源），只影响 RViz 显示，
  控制通路不消费 TF。
- QP `target_frequency=20`（`giskard_stretch_isaac.py`），低于 20 会告警；
  仿真端积分器的 `VEL_MAX_LEAD`（0.02，apartment.py）钳制目标对实测的超前量，
  同时也让接触时不会持续加压。
