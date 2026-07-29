"""An empty Isaac Sim scene: ground, lights, and nothing else.

What the Panda pick-and-place demo runs in. The apartment is the wrong backdrop
for a bolted-down arm -- its furniture is nowhere near the robot's half-metre
workspace, and its USD/URDF pair only lines up approximately, which is exactly
the ambiguity a grasp test should not have to carry. Here the only things in the
world besides the robot are the props, and both sides build those from the same
numbers.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app`
   has run -- this module imports ``isaacsim.core`` at module scope.
"""

import numpy as np
from isaacsim.core.utils import viewports
from isaacsim.core.utils.prims import create_prim, define_prim

from cram_vrb_lab.scenes.apartment.constants import GRID_USD_PATH


def load_empty_scene(world, render, camera_target=(0.45, 0.0, 0.35)):
    """Ground grid, lights, and a viewport aimed at the robot's workspace.

    :param camera_target: what the default viewport looks at -- the props, not
        the robot base, since that is where everything interesting happens.
    """
    define_prim("/World/Ground", "Xform").GetReferences().AddReference(GRID_USD_PATH)

    for index in range(4):
        create_prim(
            prim_path=f"/World/Ground/Light_{index}",
            prim_type="SphereLight",
            attributes={"inputs:intensity": 10000},
            position=(1.5 - 1.5 * index, 0, 2),
        )

    viewports.set_camera_view(
        eye=np.array([1.6, -1.4, 1.2]), target=np.array(camera_target)
    )

    for _ in range(30):
        world.step(render=render)
