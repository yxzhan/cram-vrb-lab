"""Single source of repo-relative paths for the cram_vrb_lab package.

Every module resolves assets and the CRAM submodule through these constants;
if the package ever moves within the repo, this is the only file to audit.
"""

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_DIR / "assets"
CRAM_SUBMODULE_DIR = REPO_DIR / "cognitive_robot_abstract_machine"
ROS2_WS_DIR = REPO_DIR / "ros2_ws"

DATASETS_DIR = REPO_DIR / "datasets"
"""Where recorded demonstration episodes are written.

Inside the repo rather than under ``/tmp`` because an episode is a *result* -- it
takes a run of the demo to produce and is the input to training -- but it is not
source, so it is gitignored. Overridable per run with
:data:`~cram_vrb_lab.sim.episode_recording.DATASET_ROOT_ENV`, which is how a
recording session is pointed at a disk that has room for it.
"""
