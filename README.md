# cram_isaacsim

在 Isaac Sim 中运行 [CRAM](https://github.com/cram2/cognitive_robot_abstract_machine)
（Cognitive Robot Abstract Machine）。当前演示：用 giskardpy 闭环控制公寓场景中的
Hello Robot Stretch（关节 / 末端笛卡尔 / 底盘 / 夹爪）。

| 目录 | 内容 |
|---|---|
| `examples/apartment.py` | 仿真场景 + Stretch 的 ROS2 桥 |
| `giskard_stretch/` | giskard 服务端 + `giskard_demo.ipynb` 演示 |
| `cognitive_robot_abstract_machine/` | CRAM monorepo（子模块） |
| `assets/` | USD 资产；`stretch_urdf/` 为官方 URDF 子模块 |
| `ros2_ws/` | json_msgs 消息包 |
| `binder/` | Docker 镜像定义 |

## 快速开始

```bash
git clone --recurse-submodules <this-repo>   # 环境构建见 binder/Dockerfile
source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash

# 1. 仿真
~/.local/bin/isaacsim_python_wrapper.sh examples/apartment.py
# 2. giskard 服务端（等 "giskard is ready"）
cognitive_robot_abstract_machine/.venv/bin/python giskard_stretch/giskard_stretch_isaac.py
# 3. 打开 giskard_stretch/giskard_demo.ipynb 发运动目标（内核选 "Giskard Python (venv)"）
```

详细说明与故障排查见 `giskard_stretch/README.md`。
