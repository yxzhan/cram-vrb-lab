"""The Isaac half of :mod:`cram_vrb_lab.sim.episode_recording`: captures the frames.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app` has
   run -- this module reads Isaac camera sensors and articulation views.

The shape of this file is set by one constraint: **the sim loop is a real-time
controller and must not wait for a disk.** :func:`cram_vrb_lab.sim.runner.run` steps
physics, consumes giskard's commands and republishes the state giskard closes its
loop on, all on one thread, and that thread's rate is the control rate (see
:data:`~cram_vrb_lab.sim.runner.FEEDBACK_MARGIN`). So the capture is split in two:

- On the sim thread, in :meth:`EpisodeRecorderROS.publish`: read the render products
  and the articulation, copy them, and hand the copy over. This part is unavoidable
  -- the camera's frame lives in the sim process and only this thread may read it --
  and it is why the capture rate is decoupled from the control rate.
- On :class:`_EpisodeWriter`'s own thread: encode the PNGs and append the row. This
  part is the one that would otherwise stall the loop, and nothing on it touches USD,
  PhysX or rclpy.

What connects them is a **bounded** queue, and the bound is the design decision worth
reading: see :data:`QUEUE_DEPTH`.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from std_msgs.msg import String

from cram_vrb_lab.sim.episode_recording import (
    DEFAULT_RECORD_HZ,
    RECORDING_ACK_TOPIC,
    RECORDING_TOPIC,
    dataset_root,
)
from cram_vrb_lab.sim.ros_utils import SimBridge, as_np

QUEUE_DEPTH = 120
"""Frames that may be waiting to be written before new ones are dropped.

**Bounded, and a full queue drops rather than blocks.** Blocking would be the
obvious choice and is the wrong one: the sim thread is what advances physics and
feeds giskard, so a blocked capture does not slow the recording down, it slows the
*robot* down -- and because the integrator advances the drives over measured wall
time, a slower loop means the arm really moves differently. A recording that changed
what it recorded would be worse than one with holes in it, so the holes win, and
they are counted and reported (``dropped`` in the ack and in ``meta.json``) instead
of being silent.

120 frames is twelve seconds of backlog at :data:`~cram_vrb_lab.sim.episode_recording.DEFAULT_RECORD_HZ`,
and about 72 MB of RGB at three 224x224 cameras -- enough to ride out a slow disk or
a burst of page cache writeback without being enough to matter against the memory a
loaded apartment already uses.
"""

WRITER_DRAIN_TIMEOUT = 60.0
"""Seconds :meth:`EpisodeRecorderROS._stop` waits for the writer to finish.

Long, because it is waited out on the *client's* side of a blocking stop -- the demo
is between two task attempts by then, and an episode that is half on disk is worth
more than a second saved. If it does expire the episode is still closed and said to
be short; nothing is left running.
"""

POLL_INTERVAL = 0.1
"""Seconds the writer thread waits on an empty queue before re-checking whether the
episode is closing."""


class _EpisodeWriter:
    """Writes one episode's frames to disk, on its own thread.

    Created per episode rather than once per session: an episode is a directory and a
    file handle, and tying the thread's lifetime to them means a stop cannot leave a
    row of the previous episode arriving in the next one's ``frames.jsonl``.

    Only two of its attributes cross threads, and each has exactly one writer:
    :attr:`dropped` is written by the sim thread through :meth:`offer`, and
    :attr:`written` by this thread. Both are ints, read after :meth:`close` has
    joined, so neither needs a lock.
    """

    def __init__(self, directory: Path, camera_names: List[str]):
        self.directory = directory
        self.dropped = 0
        self.written = 0
        self.error: Optional[str] = None

        directory.mkdir(parents=True, exist_ok=True)
        for name in camera_names:
            (directory / name).mkdir(exist_ok=True)
        self._frames = (directory / "frames.jsonl").open("w", buffering=1)

        self._queue: "queue.Queue[Optional[Dict]]" = queue.Queue(maxsize=QUEUE_DEPTH)
        self._closing = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="episode-writer", daemon=True
        )
        self._thread.start()

    def offer(self, record: Dict) -> bool:
        """Hand one frame over without waiting. Returns whether it was taken.

        Called from the sim thread. See :data:`QUEUE_DEPTH` for why a refusal is a
        dropped frame rather than a wait.
        """
        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        # Imported on this thread and at first use: PIL is only needed when something
        # is actually recording, and the sim must not fail to start over a dependency
        # that an idle bridge never touches.
        from PIL import Image

        while True:
            try:
                record = self._queue.get(timeout=POLL_INTERVAL)
            except queue.Empty:
                # Empty *and* closing is the only way out, so everything already
                # queued when close() was called is written before this returns.
                if self._closing.is_set():
                    return
                continue
            try:
                self._write(record, Image)
                self.written += 1
            except Exception as failure:  # noqa: BLE001 - one bad frame is not the run
                # Kept rather than raised: this thread dying silently would turn a
                # full disk into an episode that simply stops growing, and the ack
                # would still report success.
                if self.error is None:
                    self.error = f"{type(failure).__name__}: {failure}"

    def _write(self, record: Dict, Image) -> None:
        images = record.pop("images")
        index = record["i"]
        for name, rgb in images.items():
            Image.fromarray(rgb).save(self.directory / name / f"{index:06d}.png")
        self._frames.write(json.dumps(record) + "\n")

    def close(self, timeout: float = WRITER_DRAIN_TIMEOUT) -> bool:
        """Drain the queue, stop the thread, close the file. Returns whether it
        finished within ``timeout``."""
        self._closing.set()
        self._thread.join(timeout)
        drained = not self._thread.is_alive()
        self._frames.close()
        return drained


class EpisodeRecorderROS(SimBridge):
    """Records task attempts as demonstration episodes, on request.

    Idle until something starts it: one subscription, one publisher and a rate check
    per cycle, which is why :func:`cram_vrb_lab.sim.runner.build` adds it to every
    setup rather than to the ones a demo has been told will record.

    Requests are **queued** by the subscription callback and acted on in
    :meth:`apply_commands`, the same discipline
    :class:`~cram_vrb_lab.sim.scene_sync_bridge.SceneSyncROS` follows and for a
    related reason: rclpy delivers on its own thread, and starting an episode from
    there could open a writer half way through a capture that the sim thread is in
    the middle of. Handling it between two steps makes a start or a stop land on a
    frame boundary by construction.
    """

    receives_commands = True  # the subscription has to be spun

    def __init__(
        self,
        world,
        robot=None,
        integrator=None,
        cameras: Optional[Dict] = None,
        root: Optional[str] = None,
    ):
        """
        :param robot: the robot ``Articulation``; its measured joint positions are
            ``state`` and its link transforms are ``links``.
        :param integrator: the streamed-velocity integrator driving that robot. It
            holds the position targets the drives were actually given, which is
            ``action`` -- see :data:`~cram_vrb_lab.sim.episode_recording.EPISODE_LAYOUT`
            for why that and not the velocity command.
        :param cameras: ``{name: Camera}``, normally the scene's fixed rig. An empty
            mapping is allowed and records state and action alone, which is what a
            run on a machine that cannot afford the render products gets.
        :param root: where episodes are written; defaults to
            :func:`~cram_vrb_lab.sim.episode_recording.dataset_root`.
        """
        super().__init__("episode_recorder_ros")
        self.world = world
        self.robot = robot
        self.integrator = integrator
        self.cameras = dict(cameras or {})
        self.root = Path(root or dataset_root())

        self._requests: List[Dict] = []
        self._lock = threading.Lock()
        self.create_subscription(String, RECORDING_TOPIC, self._on_request, 10)
        self._ack = self.create_publisher(String, RECORDING_ACK_TOPIC, 10)

        self._writer: Optional[_EpisodeWriter] = None
        self._meta: Optional[Dict] = None
        self._index = 0
        self._period = 1.0 / DEFAULT_RECORD_HZ
        self._next_capture = 0.0
        self._started_wall = 0.0
        self._started_sim = 0.0

    # %% the switch

    def _on_request(self, message: String) -> None:
        """Queue a request. Deliberately does no capturing; see the class docstring."""
        with self._lock:
            self._requests.append(json.loads(message.data))

    def apply_commands(self, dt: float) -> None:
        """Act on every queued start/stop and acknowledge each one."""
        with self._lock:
            requests, self._requests = self._requests, []
        for request in requests:
            # Guarded for the reason SceneSyncROS guards its requests: this runs on
            # the sim loop thread, so an uncaught exception does not fail the request,
            # it ends the simulator -- and silently, because simulation_app.close()
            # beats Python to printing the traceback.
            try:
                report = self._handle(request)
            except Exception as failure:  # noqa: BLE001 - a bad request must not be fatal
                report = {
                    "id": request.get("id"),
                    "recording": self._writer is not None,
                    "error": f"{type(failure).__name__}: {failure}",
                }
                self.get_logger().error(f"episode recording failed: {report['error']}")
            self._ack.publish(String(data=json.dumps(report)))

    def _handle(self, request: Dict) -> Dict:
        command = request.get("command")
        if command == "start":
            return self._start(request)
        if command == "stop":
            return self._stop(request)
        raise ValueError(
            f"unknown episode-recording command {command!r}; expected start or stop"
        )

    def _start(self, request: Dict) -> Dict:
        episode = request.get("episode") or "episode"
        # A resend of the request that already started this episode must not open a
        # second directory -- see EpisodeRecorderClient._request on why resends
        # happen at all. Matched on the request id, not the episode name, so a demo
        # that deliberately re-records the same name still gets a fresh episode.
        if self._meta is not None and self._meta.get("request_id") == request.get("id"):
            return self._report(request, extra={"resend": True})
        if self._writer is not None:
            # Abandoned rather than refused: a demo that starts a second attempt
            # without stopping the first has abandoned it, and refusing would lose
            # every episode after the one that crashed instead of just that one.
            self._close_episode(outcome="superseded", notes={})

        stamp = time.strftime("%Y%m%dT%H%M%S")
        directory = self.root / f"{stamp}_{episode}"
        self._period = 1.0 / float(request.get("hz") or DEFAULT_RECORD_HZ)
        self._index = 0
        self._started_wall = time.time()
        self._started_sim = float(self.world.current_time)
        self._meta = {
            "request_id": request.get("id"),
            "episode": episode,
            "task": request.get("task"),
            "started_wall": self._started_wall,
            "started_sim": self._started_sim,
            "requested_hz": 1.0 / self._period,
            "cameras": sorted(self.cameras),
            "camera_resolution": self._camera_resolution(),
            "state_joints": self._state_joints(),
            "action_joints": list(getattr(self.integrator, "joint_names", []) or []),
            "link_names": self._link_names(),
        }
        self._writer = _EpisodeWriter(directory, sorted(self.cameras))
        # Written before a single frame is, so an episode whose sim dies mid-run is
        # still a directory that says what it was trying to record. Rewritten with
        # the totals at stop.
        self._write_meta()
        # Zero rather than now + period: the first capture should be the first cycle
        # after the start, not one period later, so an episode's first frame is the
        # scene the task actually began from. publish() replaces it with a real
        # deadline on that first capture.
        self._next_capture = 0.0
        print(f"[record] {directory.name}: started, task {self._meta['task']!r}, "
              f"{self._meta['requested_hz']:g} Hz, "
              f"cameras {self._meta['cameras'] or 'none'}", flush=True)
        return self._report(request)

    def _stop(self, request: Dict) -> Dict:
        if self._writer is None:
            return {"id": request.get("id"), "recording": False, "path": None}
        report = self._report(request)
        report.update(
            self._close_episode(
                outcome=request.get("outcome"), notes=request.get("notes") or {}
            )
        )
        report["recording"] = False
        return report

    def _close_episode(self, outcome: Optional[str], notes: Dict) -> Dict:
        """Flush the writer, finish ``meta.json``, and return what happened."""
        writer, self._writer = self._writer, None
        meta = self._meta or {}
        meta["outcome"] = outcome
        meta["notes"] = notes
        meta["wall_seconds"] = time.time() - self._started_wall
        meta["sim_seconds"] = float(self.world.current_time) - self._started_sim

        drained = writer.close()
        meta["frames"] = writer.written
        meta["dropped"] = writer.dropped
        meta["captured"] = self._index
        # The rate the episode was actually captured at, next to the one it asked
        # for. They differ when the sim cannot reach the requested rate -- a capture
        # can only land on a sim cycle -- and an export that resamples needs to know
        # which of the two describes the data.
        meta["achieved_hz"] = (
            self._index / meta["wall_seconds"] if meta["wall_seconds"] > 0 else 0.0
        )
        if not drained:
            meta["truncated"] = (
                f"the writer did not drain within {WRITER_DRAIN_TIMEOUT:g}s"
            )
        if writer.error:
            meta["writer_error"] = writer.error
        self._meta = meta
        self._write_meta(writer.directory)
        self._meta = None

        summary = (
            f"[record] {writer.directory.name}: {writer.written} frames"
            f" in {meta['wall_seconds']:.1f}s ({meta['sim_seconds']:.1f}s sim)"
        )
        if writer.dropped:
            summary += f", DROPPED {writer.dropped}"
        if writer.error:
            summary += f", writer error: {writer.error}"
        print(summary, flush=True)

        return {
            "path": str(writer.directory),
            "frames": writer.written,
            "dropped": writer.dropped,
            "captured": meta["captured"],
            "achieved_hz": meta["achieved_hz"],
            "wall_seconds": meta["wall_seconds"],
            "sim_seconds": meta["sim_seconds"],
            "outcome": outcome,
            "truncated": meta.get("truncated"),
            "writer_error": writer.error,
        }

    def _report(self, request: Dict, extra: Optional[Dict] = None) -> Dict:
        report = {
            "id": request.get("id"),
            "recording": self._writer is not None,
            "path": None if self._writer is None else str(self._writer.directory),
        }
        report.update(extra or {})
        return report

    def _write_meta(self, directory: Optional[Path] = None) -> None:
        target = directory or (self._writer.directory if self._writer else None)
        if target is None or self._meta is None:
            return
        (target / "meta.json").write_text(json.dumps(self._meta, indent=2) + "\n")

    # %% what an episode is made of

    def _camera_resolution(self) -> Optional[List[int]]:
        for camera in self.cameras.values():
            return list(camera.get_resolution())
        return None

    def _state_joints(self) -> List[str]:
        names = None if self.robot is None else self.robot.dof_names
        return list(names or [])

    def _link_names(self) -> List[str]:
        names = None if self.robot is None else getattr(self.robot, "body_names", None)
        return list(names or [])

    # %% the capture

    def publish(self) -> None:
        """Capture one frame, if an episode is open and one is due.

        Called by :func:`cram_vrb_lab.sim.runner.run` after ``world.step``, so what
        is read here is the state the step just produced and the frame the render
        just drew -- the same pair the robot's own bridge publishes from.

        The rate check is against the **monotonic** clock: this is an interval, and
        the deadline must not move when the wall clock does.

        The deadline is advanced **by a period from the previous deadline**, not set
        to ``now + period``, and the difference is not cosmetic. This method is only
        reached once per sim cycle, so a deadline measured from the moment of capture
        is rounded up to the next cycle every time: at a 40 ms cycle and a 100 ms
        period that is a capture every 120 ms, i.e. 8.3 Hz where the episode's
        ``meta.json`` claims 10 -- measured, 67 frames in 8.0 s. Advancing the
        deadline instead keeps the *average* exact (the rounding alternates 120/80
        against a deadline that does not drift) and leaves only the one-cycle jitter,
        which is what the two timestamps are recorded for.

        ``max(now, ...)`` is what stops that becoming a catch-up burst: after a hitch
        the deadline is pulled up to the present rather than firing once per cycle
        until it has caught up, so a slow patch costs late frames and never a run of
        frames closer together than the sim can actually produce.
        """
        if self._writer is None:
            return
        now = time.monotonic()
        if now < self._next_capture:
            return
        self._next_capture = max(now, self._next_capture + self._period)

        try:
            record = self._capture()
        except Exception as failure:  # noqa: BLE001 - see apply_commands
            self.get_logger().error(f"episode capture failed: {failure}")
            return
        if record is None:
            return
        self._writer.offer(record)
        self._index += 1

    def _capture(self) -> Optional[Dict]:
        """Read one frame's worth of everything, as plain numpy the writer owns.

        Every array is copied here, on the sim thread. The camera buffers and the
        articulation's tensors are reused by the next step, so handing the writer a
        view would mean writing whatever the robot was doing by the time the PNG was
        encoded -- a dataset that is subtly wrong rather than obviously broken.
        """
        images = {}
        for name, camera in self.cameras.items():
            rgba = camera.get_rgba()
            # An empty array is what a render product answers before it has produced
            # a frame. Skipped rather than recorded black, so an export can tell a
            # missing image from a dark room.
            if rgba is None or len(rgba) == 0:
                continue
            images[name] = np.ascontiguousarray(as_np(rgba)[..., :3], dtype=np.uint8)

        record = {
            "i": self._index,
            "sim_t": float(self.world.current_time),
            "wall_t": time.time(),
            "images": images,
        }

        if self.robot is not None:
            record["state"] = [
                round(float(value), 6)
                for value in as_np(self.robot.get_joint_positions())[0]
            ]
            view = getattr(self.robot, "_physics_view", None)
            if view is not None:
                transforms = as_np(view.get_link_transforms()).reshape(-1, 7)
                record["links"] = [
                    [round(float(value), 6) for value in row] for row in transforms
                ]

        targets = getattr(self.integrator, "targets", None)
        record["action"] = (
            None if targets is None else [round(float(value), 6) for value in targets]
        )
        return record

    # %% shutdown

    def destroy_node(self) -> None:
        """Close an open episode before the node goes away.

        The sim is taken down by ``runner.run``'s ``finally``, which a
        ``KeyboardInterrupt`` reaches on every ordinary exit -- so without this,
        stopping a demo with Ctrl-C would leave the last episode's queued frames
        unwritten and its ``meta.json`` without a frame count.
        """
        if self._writer is not None:
            self._close_episode(outcome="interrupted", notes={})
        super().destroy_node()
