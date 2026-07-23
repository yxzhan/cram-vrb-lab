"""Single source of repo-relative paths for the cram_vrb_lab package.

Every module resolves assets and the CRAM submodule through these constants;
if the package ever moves within the repo, this is the only file to audit.
"""

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_DIR / "assets"
CRAM_SUBMODULE_DIR = REPO_DIR / "cognitive_robot_abstract_machine"
