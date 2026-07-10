# isaacsim-giskard

在 Isaac Sim 中用 [giskardpy](https://github.com/cram2/cognitive_robot_abstract_machine)
（闭环 QP 全身运动控制）控制 Hello Robot Stretch 的最小演示仓库。

```
Isaac Sim (examples/apartment.py)                     giskard 服务端
  /stretch/joint_states       ────────────────────►  关节状态同步
  /odom                       ────────────────────►  底盘里程计同步 (DiffDrive)
  /stretch/cmd_vel            ◄────────────────────  底盘 Twist（闭环, 带 1s 看门狗）
  /stretch/joint_velocity_cmd ◄────────────────────  关节速度流（仿真内积分为位置目标）

  giskard_demo.ipynb ── JsonAction(/giskard/command) ──► giskard
```

## 仓库结构

| 目录 | 内容 |
|---|---|
| `examples/apartment.py` | 仿真：公寓场景 + Stretch + ROS2 桥（含速度积分与 cmd_vel 看门狗） |
| `giskard_stretch/` | giskard 服务端配置 + **`giskard_demo.ipynb`** 演示 notebook |
| `cognitive_robot_abstract_machine/` | giskardpy monorepo（git 子模块） |
| `assets/` | 公寓场景与 Stretch 的 USD 资产；`stretch_urdf/` 为官方 URDF 子模块 |
| `ros2_ws/` | json_msgs 消息包（giskard 的 action 接口） |
| `binder/` | Docker 镜像定义（Isaac Sim + ROS2 Jazzy + giskard 环境） |

## 获取代码

```bash
git clone --recurse-submodules <this-repo>
```

## 环境准备（Docker 镜像已内置；本机手动执行如下）

```bash
# 1. ROS 依赖
sudo apt-get install -y ros-jazzy-rclpy-message-converter ros-jazzy-py-trees ros-jazzy-py-trees-ros
# 2. giskard python 环境（uv workspace）
cd cognitive_robot_abstract_machine
uv sync
.venv/bin/pip install py_trees argcomplete
.venv/bin/python -m ipykernel install --user --name giskard --display-name "Giskard Python (venv)"
cd ..
# 3. json_msgs 消息包
source /opt/ros/jazzy/setup.bash
cd ros2_ws && colcon build --packages-select json_msgs && cd ..
```

## 运行

每个终端先 source：

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
```

1. **仿真**（Isaac Sim python）：
   ```bash
   ~/.local/bin/isaacsim_python_wrapper.sh examples/apartment.py
   ```
2. **giskard 服务端**（等待日志 `giskard is ready`）：
   ```bash
   cognitive_robot_abstract_machine/.venv/bin/python giskard_stretch/giskard_stretch_isaac.py
   ```
3. **发运动目标**：打开 `giskard_stretch/giskard_demo.ipynb`（内核选
   "Giskard Python (venv)"），依次执行关节 / 夹爪 / 末端笛卡尔 / 底盘的演示 cell。

精度、调参与故障排查见 `giskard_stretch/README.md`。
