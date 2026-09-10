# %% [markdown]
# ## Launch

# %%
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from IPython import get_ipython
in_notebook = get_ipython().__class__.__name__ == "ZMQInteractiveShell"

REPO =  Path.cwd().resolve().parent if in_notebook else Path.cwd().resolve()
sys.path.insert(0, str(REPO))

os.environ.setdefault("ISAAC_HEADLESS", "1")
os.environ.setdefault("ISAAC_LIVESTREAM", "1")

# os.environ["ISAAC_WINDOW"] = "1920x1080"
# os.environ["ISAAC_WINDOW"] = "1280x720"
# os.environ["ISAAC_WINDOW"] = "960x540"
# os.environ["ISAAC_WINDOW"] = "854x480"
# os.environ["ISAAC_WINDOW"] = "768x432"
os.environ["ISAAC_WINDOW"] = "640x360"
# os.environ["ISAAC_WINDOW"] = "512x288"
os.environ["DISPLAY"] = ":1"


# Put the four kitchen objects -- cup, bowl, cereal box, milk box -- on the cabinet worktop
# os.environ["ISAAC_KITCHEN_PROPS"] = "1"
os.environ["ISAAC_KITCHEN_PROPS"] = "0"

RVIZ_CONFIG = REPO / "demos" / "rviz" / "garmi.rviz"
ROBOT, SCENE = "garmi", "garmi_apartment"
SPAWN_POSITION = (0, 5.0, 0.0259)
SPAWN_YAW = math.pi / 2

from launcher import (
    start_giskard_server,
    start_isaac_sim,
    start_rviz,
    start_streaming_client,
    stop,
)
from cram_vrb_lab.sim.isaac_app import livestream_enabled

rviz_proc = start_rviz(rviz_config=RVIZ_CONFIG)
sim_proc = start_isaac_sim(robot=ROBOT, scene=SCENE, camera="none",
                           spawn_position=SPAWN_POSITION, spawn_yaw=SPAWN_YAW)
stream_proc = start_streaming_client() if livestream_enabled() else None
giskard_proc = start_giskard_server(robot=ROBOT, scene=SCENE, control_hz=15,
                                    spawn_position=SPAWN_POSITION, spawn_yaw=SPAWN_YAW)

from time import sleep
while True:
    sleep(1)
