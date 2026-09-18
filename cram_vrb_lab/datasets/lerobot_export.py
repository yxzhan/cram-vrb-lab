"""Turn recorded episodes into a LeRobot dataset, offline.

Run from the repo root, in the CRAM venv -- the only interpreter here with
``lerobot`` in it::

    binder/cram_python_wrapper.sh -m cram_vrb_lab.datasets.lerobot_export \\
        --repo-id cram_vrb_lab/garmi_transport --out datasets/lerobot

Input is whatever :mod:`cram_vrb_lab.sim.episode_recording` wrote: one directory
per task attempt, described by
:data:`~cram_vrb_lab.sim.episode_recording.EPISODE_LAYOUT`. Output is a LeRobot
dataset the training code reads.

**Offline, and a separate process, on purpose.** The sim writes PNGs and one JSON
object per frame because that is all it can afford to do inside a real-time control
cycle; everything that needs a decision, a second pass over the data, or four
gigabytes of torch happens here instead. That also means a bad export costs an
export rather than a run of the robot -- the episodes on disk are the artefact, and
this can be re-run against them as often as the questions below get better answers.

The one thing this must do, and the reason it cannot be a file copy, is
**resampling**. See :func:`resample`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_FPS = 10.0
"""Frames per second of the exported dataset.

Matches :data:`~cram_vrb_lab.sim.episode_recording.DEFAULT_RECORD_HZ`, so the common
case resamples onto a grid the data was already nearly on and every output frame has
a source within a few milliseconds. Export at a *higher* rate than the recording and
every frame is still real -- it is just that consecutive output frames start
repeating the same source.
"""

IMAGE_PREFIX = "observation.images"
"""Feature-name prefix LeRobot expects for camera streams."""


@dataclass
class Episode:
    """One recorded attempt, loaded into memory.

    The images are deliberately *not* loaded here -- an episode is a few hundred
    frames times three cameras, and the export only ever needs the ones resampling
    selected.
    """

    directory: Path
    meta: Dict
    rows: List[Dict]

    @property
    def task(self) -> str:
        return self.meta.get("task") or "unknown task"

    @property
    def outcome(self) -> Optional[str]:
        return self.meta.get("outcome")

    @property
    def cameras(self) -> List[str]:
        return list(self.meta.get("cameras") or [])

    @property
    def wall_times(self) -> np.ndarray:
        return np.array([row["wall_t"] for row in self.rows], dtype=float)


@dataclass
class ResampleReport:
    """What resampling had to do to an episode, so that it can be looked at.

    Printed per episode rather than summed, because the numbers that matter are the
    worst ones: an export whose *average* offset is 3 ms can still contain a
    two-second hole where the writer dropped frames, and only :attr:`max_offset` and
    :attr:`late` say so.
    """

    frames: int
    """Output frames written."""

    max_offset: float
    """Seconds between the furthest output frame and the source it was given."""

    late: int
    """Output frames whose nearest source is more than half a period away, i.e. the
    ones sitting in a gap in the recording rather than next to a real sample."""

    reused: int
    """Output frames that repeat the previous frame's source, because the grid is
    finer than the recording was at that moment."""

    unused: int
    """Recorded frames no output frame selected, because the grid is coarser."""


@dataclass
class ExportReport:
    """The whole export, for the caller and for the final printout."""

    episodes: int = 0
    frames: int = 0
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    per_episode: List[Tuple[str, ResampleReport]] = field(default_factory=list)


def load_episode(directory: Path) -> Episode:
    """Read one episode directory's ``meta.json`` and ``frames.jsonl``.

    :raises FileNotFoundError: if either is missing, which is what an interrupted
        recording looks like from here.
    """
    meta_path = directory / "meta.json"
    frames_path = directory / "frames.jsonl"
    if not meta_path.exists() or not frames_path.exists():
        raise FileNotFoundError(
            f"{directory} is not an episode: expected meta.json and frames.jsonl"
        )
    meta = json.loads(meta_path.read_text())
    with frames_path.open() as handle:
        # A line at a time, tolerating a truncated last one: the sim writes this
        # line-buffered as it goes, so an episode whose process was killed mid-write
        # ends in a partial object. Everything before it is still good data.
        rows = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return Episode(directory=directory, meta=meta, rows=rows)


def resample(times: np.ndarray, fps: float) -> Tuple[np.ndarray, ResampleReport]:
    """Map a uniform grid at ``fps`` onto the recorded frames, nearest-neighbour.

    **Why this exists.** The recording is not uniform and cannot be made uniform: a
    capture lands on a sim cycle, and the sim's cycle is a real-time control loop
    whose period moves with load -- measured at 39 to 126 ms against a 100 ms target
    in an eight-second episode. LeRobot carries a single ``fps`` for the whole dataset
    and assumes the frames are evenly spaced in time, so something has to reconcile
    the two, and doing it here rather than in the sim is what keeps the recording
    honest about what actually happened.

    **Why nearest-neighbour rather than interpolation**, which would give a smoother
    state trajectory: an output frame has to be internally consistent. The image and
    the joint positions in one frame must be the same instant, or a policy is trained
    to predict an action from an observation that never existed. Images cannot be
    interpolated, so interpolating the state would pair a synthesised state with a
    real image from up to half a period away. Snapping everything to one source frame
    costs up to half a period of timing error and buys a frame that is a thing that
    happened.

    The grid starts at the first recorded frame and steps by ``1 / fps`` for as long
    as the episode lasts, so an episode's first output frame is always its first real
    one -- the scene the attempt began from.

    :return: the chosen source index per output frame, and what it cost.
    """
    if len(times) == 0:
        return np.array([], dtype=int), ResampleReport(0, 0.0, 0, 0, 0)
    if len(times) == 1:
        return np.zeros(1, dtype=int), ResampleReport(1, 0.0, 0, 0, 0)

    period = 1.0 / fps
    span = float(times[-1] - times[0])
    grid = times[0] + np.arange(int(math.floor(span / period)) + 1) * period

    # The frame after each grid point, then whichever of it and its predecessor is
    # closer. searchsorted rather than a distance matrix: an episode is thousands of
    # frames and this is O(n log n) instead of O(n^2).
    after = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
    before = after - 1
    chosen = np.where(
        grid - times[before] <= times[after] - grid, before, after
    )
    offsets = np.abs(times[chosen] - grid)

    return chosen, ResampleReport(
        frames=len(chosen),
        max_offset=float(offsets.max()),
        late=int((offsets > period / 2).sum()),
        reused=int((np.diff(chosen) == 0).sum()),
        unused=len(times) - len(set(chosen.tolist())),
    )


def action_for(row: Dict, state_joints: Sequence[str],
               action_joints: Sequence[str]) -> List[float]:
    """The action for one frame, filling in the frames that have none.

    ``action`` is ``null`` until giskard's first command reaches the integrator, and
    again after a scene reset drops its targets. Those frames are not missing data:
    the drives were holding the position they were already at, so the target they
    were given *is* the measured position of those joints. That is what this
    substitutes, which is why the recorder writes both joint-name lists into
    ``meta.json`` -- without them the two vectors could not be lined up.

    Dropping such frames instead was the alternative and is worse: they are the start
    of the episode, i.e. exactly the initial conditions a policy has to learn to act
    from.
    """
    action = row.get("action")
    if action is not None:
        return action
    index = {name: position for position, name in enumerate(state_joints)}
    state = row["state"]
    return [float(state[index[name]]) for name in action_joints]


def build_features(meta: Dict, cameras: Sequence[str], use_videos: bool,
                   ee_link: Optional[str] = None) -> Dict:
    """The LeRobot feature dict for episodes recorded with this ``meta``.

    The joint names go into ``names`` rather than being thrown away: a dataset whose
    state is 26 anonymous floats cannot be lined up against a robot again, and this
    is the only place that mapping is still known.
    """
    height, width = (meta.get("camera_resolution") or [224, 224])[::-1]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(meta["state_joints"]),),
            "names": list(meta["state_joints"]),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(meta["action_joints"]),),
            "names": list(meta["action_joints"]),
        },
    }
    for camera in cameras:
        features[f"{IMAGE_PREFIX}.{camera}"] = {
            # "video" makes LeRobot encode the frames into one file per camera per
            # episode, which is ~50x smaller than the PNGs and is what its own
            # default does; "image" keeps them as files, which needs no encoder.
            "dtype": "video" if use_videos else "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    if ee_link is not None:
        features["observation.ee_pose"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
        }
    return features


def _episode_directories(root: Path) -> List[Path]:
    return sorted(
        entry for entry in root.iterdir()
        if entry.is_dir() and (entry / "meta.json").exists()
    )


def _check_compatible(first: Dict, meta: Dict, cameras: Sequence[str]) -> Optional[str]:
    """Why these two episodes cannot go into one dataset, or ``None``.

    Checked up front rather than left to LeRobot, which discovers the mismatch part
    way through writing and leaves a half-built dataset behind. The usual cause is
    mundane and worth naming in the message: episodes recorded before and after
    someone changed the camera rig.
    """
    if list(meta["state_joints"]) != list(first["state_joints"]):
        return "different state joints"
    if list(meta["action_joints"]) != list(first["action_joints"]):
        return "different action joints"
    if list(meta.get("cameras") or []) != list(cameras):
        return (
            f"different cameras ({meta.get('cameras')} vs {list(cameras)})"
        )
    if (meta.get("camera_resolution") or []) != (first.get("camera_resolution") or []):
        return "different camera resolution"
    return None


def export(
    episode_dirs: Sequence[Path],
    repo_id: str,
    out: Path,
    fps: float = DEFAULT_FPS,
    outcomes: Optional[Sequence[str]] = None,
    use_videos: bool = True,
    ee_link: Optional[str] = None,
    robot_type: str = "garmi",
) -> ExportReport:
    """Build one LeRobot dataset out of ``episode_dirs``.

    :param outcomes: only export episodes whose ``meta["outcome"]`` is one of these;
        ``None`` exports every one. The recorder deliberately keeps failed attempts
        -- whether to train on them is this function's caller's question, not the
        robot's.
    :param use_videos: encode each camera to video rather than writing PNGs.
    :param ee_link: a link name from ``meta["link_names"]`` to export as
        ``observation.ee_pose``. Off by default, because which link is the
        end-effector depends on the arm a run used.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    report = ExportReport()
    dataset = None
    first_meta: Optional[Dict] = None
    cameras: List[str] = []
    link_index: Optional[int] = None

    for directory in episode_dirs:
        episode = load_episode(directory)
        if not episode.rows:
            report.skipped.append((directory.name, "no frames"))
            continue
        if outcomes is not None and episode.outcome not in outcomes:
            report.skipped.append((directory.name, f"outcome {episode.outcome!r}"))
            continue

        if dataset is None:
            first_meta = episode.meta
            cameras = episode.cameras
            if ee_link is not None:
                names = list(episode.meta.get("link_names") or [])
                if ee_link not in names:
                    raise SystemExit(
                        f"--ee-link {ee_link!r} is not a link of these episodes. "
                        f"Available: {', '.join(names) or '(none recorded)'}"
                    )
                link_index = names.index(ee_link)
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=int(round(fps)),
                features=build_features(episode.meta, cameras, use_videos, ee_link),
                root=out,
                robot_type=robot_type,
                use_videos=use_videos,
            )
        else:
            incompatible = _check_compatible(first_meta, episode.meta, cameras)
            if incompatible:
                report.skipped.append((directory.name, incompatible))
                continue

        chosen, resampled = resample(episode.wall_times, fps)
        state_joints = episode.meta["state_joints"]
        action_joints = episode.meta["action_joints"]

        for source in chosen:
            row = episode.rows[int(source)]
            frame = {
                "observation.state": np.asarray(row["state"], dtype=np.float32),
                "action": np.asarray(
                    action_for(row, state_joints, action_joints), dtype=np.float32
                ),
                "task": episode.task,
            }
            for camera in cameras:
                path = directory / camera / f"{row['i']:06d}.png"
                if not path.exists():
                    # The writer emits a row and its images together, so a row
                    # without its PNG means the directory was edited or the disk
                    # filled. Loud, because the alternative is a dataset that is
                    # silently missing the frames nobody noticed.
                    raise SystemExit(
                        f"{path} is missing but {directory.name}/frames.jsonl has "
                        f"frame {row['i']}. The episode is incomplete; delete it or "
                        "export the others by name."
                    )
                frame[f"{IMAGE_PREFIX}.{camera}"] = np.asarray(
                    Image.open(path).convert("RGB"), dtype=np.uint8
                )
            if link_index is not None:
                frame["observation.ee_pose"] = np.asarray(
                    row["links"][link_index], dtype=np.float32
                )
            dataset.add_frame(frame)

        dataset.save_episode()
        report.episodes += 1
        report.frames += resampled.frames
        report.per_episode.append((directory.name, resampled))
        print(
            f"  {directory.name}: {resampled.frames} frames"
            f" from {len(episode.rows)} recorded,"
            f" max offset {resampled.max_offset * 1e3:.0f} ms"
            + (f", {resampled.late} in gaps" if resampled.late else "")
            + (f", {resampled.reused} repeated" if resampled.reused else "")
            + (f", {resampled.unused} dropped" if resampled.unused else "")
            + f"  [{episode.outcome}]",
            flush=True,
        )

    if dataset is not None:
        # Without this the parquet footers are never written and the dataset cannot
        # be loaded again -- LeRobotDataset.finalize's own docstring says so, and
        # nothing raises if it is skipped. The failure surfaces only when something
        # tries to read the dataset, by which time the run that produced it is over.
        dataset.finalize()
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    from cram_vrb_lab.sim.episode_recording import dataset_root

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--episodes", default=None,
        help="directory holding the recorded episodes (default: where the sim "
             "writes them, i.e. $CRAM_VRB_LAB_DATASET_ROOT or ./datasets).",
    )
    parser.add_argument(
        "--repo-id", default="cram_vrb_lab/garmi",
        help="LeRobot dataset id, '<owner>/<name>' (default: %(default)s). Nothing "
             "is uploaded; it is the dataset's name.",
    )
    parser.add_argument(
        "--out", default=None,
        help="where to write the dataset (default: <episodes>/lerobot/<name>).",
    )
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help="frames per second to resample onto (default: %(default)s).",
    )
    parser.add_argument(
        "--outcome", action="append", default=None, metavar="LABEL",
        help="only export episodes with this outcome; repeatable. Default: all of "
             "them. The labels demos/garmi_demo.py writes are success, partial, "
             "failed and timeout.",
    )
    parser.add_argument(
        "--images", action="store_true",
        help="write PNGs instead of encoding video. Bigger by roughly 50x; for when "
             "the video encoder is the thing being worked around.",
    )
    parser.add_argument(
        "--ee-link", default=None, metavar="NAME",
        help="also export this link's pose as observation.ee_pose.",
    )
    parser.add_argument(
        "--robot-type", default="garmi",
        help="recorded into the dataset's metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="list the episodes that would be exported, and stop.",
    )
    args = parser.parse_args(argv)

    episodes_root = Path(args.episodes or dataset_root())
    if not episodes_root.is_dir():
        raise SystemExit(f"no episodes directory at {episodes_root}")
    directories = _episode_directories(episodes_root)
    if not directories:
        raise SystemExit(
            f"no episodes under {episodes_root} -- an episode is a directory with a "
            "meta.json in it. Has a demo run with RECORD_EPISODES on?"
        )

    if args.list:
        for directory in directories:
            meta = json.loads((directory / "meta.json").read_text())
            print(f"{directory.name}  {meta.get('frames', '?'):>5} frames  "
                  f"{str(meta.get('outcome')):<12} {meta.get('task')!r}")
        return 0

    # Under the episodes root, not beside it: that whole tree is gitignored, and an
    # export dropped one level up lands in the repo. A directory with no meta.json
    # in it is not an episode, so _episode_directories skips this on the next run.
    out = Path(args.out) if args.out else (
        episodes_root / "lerobot" / args.repo_id.split("/")[-1]
    )
    print(f"exporting {len(directories)} episode(s) from {episodes_root}")
    print(f"        -> {out} as {args.repo_id} at {args.fps:g} fps")
    report = export(
        directories,
        repo_id=args.repo_id,
        out=out,
        fps=args.fps,
        outcomes=args.outcome,
        use_videos=not args.images,
        ee_link=args.ee_link,
        robot_type=args.robot_type,
    )
    print(f"\n{report.episodes} episode(s), {report.frames} frames -> {out}")
    for name, reason in report.skipped:
        print(f"  skipped {name}: {reason}")
    if report.episodes == 0:
        print("nothing was exported.")
        return 1
    worst = max(resampled.max_offset for _, resampled in report.per_episode)
    gaps = sum(resampled.late for _, resampled in report.per_episode)
    print(f"worst resampling offset {worst * 1e3:.0f} ms"
          + (f"; {gaps} frame(s) landed in gaps in the recording" if gaps
             else "; no frame landed in a gap"))
    print("\nRead it back with:")
    print(f'  LeRobotDataset("{args.repo_id}", root="{out}", revision="local")')
    print("  -- revision= keeps it off the Hugging Face Hub, which __init__ "
          "otherwise contacts even for a local dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
