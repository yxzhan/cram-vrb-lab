"""Record one task attempt as one demonstration episode, on demand.

The demo drives the switch and Isaac does the recording::

    from cram_vrb_lab.sim.episode_recording import EpisodeRecorderClient

    recorder = EpisodeRecorderClient(node)
    recorder.start(task="put the bowl on the dining table", episode="run_001")
    ...                                       # the plan runs
    print(recorder.stop(outcome="success"))   # blocks until the sim has flushed

**Why the switch crosses ROS and the frames do not.** The two halves of a recording
have opposite shapes. The switch is two words a couple of times a minute, and it has
to come from the process that knows when a task attempt begins and ends -- which is
the demo, in the CRAM venv, not the sim. The frames are 224x224x3 three times per
capture, and they are already in the sim's own memory; sending them over ROS would
mean a ``tobytes`` copy plus an rclpy serialisation of every one, inside the
single-threaded loop that giskard's control rate depends on. ``publish_tf`` costs
6 ms of a 32 ms cycle to publish 65 *tiny* transforms
(:data:`~cram_vrb_lab.robots.garmi.isaac_node.TF_PUBLISH_HZ` exists because of it),
so the frames are written to disk by a thread on the sim side and only the switch
travels. See ``docs/vla-data-collection-assessment.md``.

A topic pair rather than a service, for the reason
:mod:`cram_vrb_lab.sim.scene_sync` gives: a service carrying a task instruction and
an outcome needs an interface package built into ``ros2_ws``, and JSON on a
``std_msgs/String`` with an ack keyed by request id is the shape this repo already
uses to cross the same gap. The client still behaves like a service call --
:meth:`EpisodeRecorderClient.start` and :meth:`~EpisodeRecorderClient.stop` block
until their own ack comes back.

**What an episode is on disk** is described by :data:`EPISODE_LAYOUT`; nothing here
writes it (that is :mod:`cram_vrb_lab.sim.episode_recorder_bridge`) and nothing here
reads it either. Building a LeRobot dataset out of a directory of these is a
separate, offline step in an interpreter that has ``lerobot`` in it -- which neither
of this repo's two has, deliberately.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, List, Optional

from std_msgs.msg import String

from cram_vrb_lab.paths import DATASETS_DIR

RECORDING_TOPIC = "/cram_vrb_lab/episode_recording"
"""Start/stop requests, as JSON. See :func:`encode_request`."""

RECORDING_ACK_TOPIC = "/cram_vrb_lab/episode_recording_ack"
"""What the sim did with each request, as JSON, keyed by its ``id``."""

DATASET_ROOT_ENV = "CRAM_VRB_LAB_DATASET_ROOT"
"""Environment variable overriding where episodes are written.

Read in the **sim** process, since that is what does the writing -- so it is set
next to ``ISAAC_FIXED_CAMERAS`` at the top of a demo, and ``launcher.start_isaac_sim``
inherits it into the subprocess.
"""

DEFAULT_RECORD_HZ = 10.0
"""Captures per second, when a request names no rate of its own.

**Deliberately below the control rate**, which is the first of the three cost
reductions the assessment lists: the sim's cycle runs at ~20 Hz with the fixed
camera rig loaded and every one of those cycles feeds giskard, so capturing on each
of them would double the render readback for frames a policy does not need. 10 Hz is
also a sane LeRobot ``fps`` to resample onto.

It is a *wall-clock* rate. Both clocks are stamped on every frame and neither is
uniform in the other -- see :data:`EPISODE_LAYOUT` -- so the resampling that makes a
fixed-fps dataset out of this happens at export time, not here.
"""

RESEND_INTERVAL = 1.0
"""Seconds between resends of an unacknowledged request; see
:meth:`EpisodeRecorderClient._request` for why resending is needed and safe."""

DEFAULT_TIMEOUT = 30.0
"""Seconds to wait for an ack.

Generous because of :meth:`EpisodeRecorderClient.stop`, which is answered only once
the writer thread has drained its backlog to disk -- a start is a directory
creation, but a stop can be a second of queued PNGs.
"""

EPISODE_LAYOUT = """
<root>/<stamp>_<episode>/
    meta.json        written at start and rewritten at stop
    frames.jsonl     one JSON object per captured frame, in capture order
    <camera>/000000.png, 000001.png, ...   one directory per camera
"""
"""What :mod:`cram_vrb_lab.sim.episode_recorder_bridge` writes, and what an offline
export reads.

``meta.json`` carries what is constant over the episode: the task instruction, the
joint names behind ``state`` and ``action``, the body names behind ``links``, the
camera names and resolution, the requested rate, and -- once the episode is stopped
-- the outcome, the frame count and how many frames were dropped.

``frames.jsonl`` carries one object per frame:

``i``
    Frame index. **The PNG filenames use it too, and it can have gaps**: a frame the
    writer could not keep up with is dropped whole (see
    :data:`~cram_vrb_lab.sim.episode_recorder_bridge.QUEUE_DEPTH`), so join the
    images to the rows on this number rather than assuming row *n* is image *n*.
``sim_t``
    Seconds of simulated time since the sim started. Uniform by construction: every
    cycle advances it by exactly ``rendering_dt``.
``wall_t``
    Unix time at capture. **Not** uniform in ``sim_t``, and the gap between the two
    is not bookkeeping -- the integrator advances the drives over measured wall time
    (:meth:`~cram_vrb_lab.sim.velocity_integrator.StreamedVelocityIntegrator.step`),
    so a loaded machine really does move the arm further per simulated step. Both are
    recorded because only both together say what the robot did.
``state``
    Measured joint positions, every DOF of the articulation, ordered by
    ``meta["state_joints"]``.
``action``
    The position targets the drives were actually given, ordered by
    ``meta["action_joints"]``, or ``null`` before the first command arrives. **Not
    the velocity command giskard sent**: the integrator clamps the target to the
    measured position plus or minus
    :data:`~cram_vrb_lab.sim.velocity_integrator.MAX_LEAD` and snaps it on the
    transition into zero velocity, so the applied target is not the integral of the
    commanded velocity, and a policy trained on velocities would have to learn that
    clamp before it could reproduce the motion.
``links``
    Every link's pose in the world frame, ``[x, y, z, qx, qy, qz, qw]`` per body,
    ordered by ``meta["link_names"]``. The end-effector pose is in here rather than
    singled out, because which link that is depends on the arm a run used and
    reading all of them costs one call.
"""


def encode_request(
    command: str,
    request_id: str,
    task: Optional[str] = None,
    episode: Optional[str] = None,
    hz: Optional[float] = None,
    outcome: Optional[str] = None,
    notes: Optional[Dict] = None,
) -> str:
    """Build the JSON one start/stop request carries.

    :param command: ``"start"`` or ``"stop"``.
    :param task: the natural-language instruction the episode demonstrates. Required
        by a VLA and by nothing else in this repo, which is exactly why it has to be
        recorded at the moment the episode is: it is the one field no amount of
        replaying the data can reconstruct.
    :param episode: a name for the attempt, e.g. ``"run_003"``. The sim prefixes a
        timestamp, so a name that repeats across sessions still lands in its own
        directory.
    :param hz: captures per second; :data:`DEFAULT_RECORD_HZ` when omitted.
    :param outcome: how the attempt ended, on a stop. Free text -- the demo's own
        verdict, e.g. ``"success"``, ``"missed"``, ``"timeout"``. Recorded rather
        than acted on: a failed attempt is still data, and whether to train on it is
        a question for the export.
    :param notes: anything else worth keeping with the episode, merged into
        ``meta.json``. For the numbers a demo knows and the sim cannot, such as how
        far an object ended up from its target.
    """
    return json.dumps(
        {
            "id": request_id,
            "command": command,
            "task": task,
            "episode": episode,
            "hz": hz,
            "outcome": outcome,
            "notes": notes or {},
        }
    )


def dataset_root() -> str:
    """Where the sim writes episodes: :data:`DATASET_ROOT_ENV`, or
    :data:`~cram_vrb_lab.paths.DATASETS_DIR`."""
    return os.environ.get(DATASET_ROOT_ENV) or str(DATASETS_DIR)


class EpisodeRecorderClient:
    """Starts and stops recordings in the sim, and waits for each to be acknowledged.

    :param node: an rclpy node that is already being spun; the ack arrives on that
        node's executor, so a client on an unspun node will always time out.
    """

    def __init__(self, node, timeout: float = DEFAULT_TIMEOUT):
        self.node = node
        self.timeout = timeout
        self._acks: Dict[str, Dict] = {}
        self._publisher = node.create_publisher(String, RECORDING_TOPIC, 10)
        self._subscription = node.create_subscription(
            String, RECORDING_ACK_TOPIC, self._on_ack, 10
        )

    def _on_ack(self, message: String) -> None:
        ack = json.loads(message.data)
        self._acks[ack["id"]] = ack

    def _request(self, timeout: Optional[float] = None, **fields) -> Dict:
        """Publish one request and block until its own ack comes back.

        :raises TimeoutError: if no ack arrives, which means the sim is not running
            this bridge rather than that the request half-applied.
        :raises RuntimeError: if the sim answered but reported an error.
        """
        request_id = str(uuid.uuid4())
        payload = encode_request(request_id=request_id, **fields)
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)

        # Wait for the bridge's subscription to be matched before publishing, for the
        # reason SceneSyncClient.apply documents at length: a topic is not a queue,
        # and a message published before discovery has paired the two ends is dropped
        # silently. That would lose the *first* episode of every session.
        while time.monotonic() < deadline:
            if self._publisher.get_subscription_count() > 0:
                break
            time.sleep(0.02)

        # Resending is safe because a request names the state it wants -- recording
        # or not recording -- rather than a change to make, and the bridge ignores a
        # start while it is already recording that episode. See _on_request there.
        last_sent = 0.0
        while time.monotonic() < deadline:
            if request_id in self._acks:
                ack = self._acks.pop(request_id)
                if ack.get("error"):
                    raise RuntimeError(f"episode recording failed: {ack['error']}")
                return ack
            now = time.monotonic()
            if now - last_sent > RESEND_INTERVAL:
                self._publisher.publish(String(data=payload))
                last_sent = now
            time.sleep(0.01)
        raise TimeoutError(
            f"no episode-recording ack in {self.timeout:g}s -- is the sim running "
            f"with the EpisodeRecorderROS bridge, and is {RECORDING_ACK_TOPIC} "
            "reaching this node?"
        )

    def start(
        self,
        task: str,
        episode: Optional[str] = None,
        hz: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Dict:
        """Begin recording; returns the sim's report, including ``path``.

        A start while an episode is already open closes that one first, with the
        outcome ``"superseded"``. That is the honest reading of what happened -- a
        demo that starts a second attempt without stopping the first has abandoned
        it -- and it means a crashed run costs one episode rather than every episode
        after it.
        """
        return self._request(
            command="start", task=task, episode=episode, hz=hz, timeout=timeout
        )

    def stop(
        self,
        outcome: Optional[str] = None,
        notes: Optional[Dict] = None,
        timeout: Optional[float] = None,
    ) -> Dict:
        """Close the open episode and wait for it to reach disk.

        Returns the sim's report: ``path``, ``frames`` written, ``dropped``, and the
        episode's duration on both clocks. A stop with nothing recording is not an
        error -- it reports ``recording: false`` and no path -- so a demo's cleanup
        path can call it unconditionally.
        """
        return self._request(
            command="stop", outcome=outcome, notes=notes, timeout=timeout
        )


def describe(report: Dict) -> str:
    """One line summarising what :meth:`EpisodeRecorderClient.stop` returned.

    Here rather than in the demo because the interesting part is the part that is
    easy not to print: ``dropped``. A dropped frame is not an error and nothing
    raises on it, but an episode that dropped a third of its frames is an episode
    whose action sequence has holes in it, and the number belongs wherever the
    episode is mentioned.
    """
    if not report.get("path"):
        return "nothing was recording"
    line = (
        f"{report['path']}: {report.get('frames', 0)} frames"
        f" in {report.get('wall_seconds', 0.0):.1f}s"
        f" ({report.get('sim_seconds', 0.0):.1f}s sim)"
    )
    dropped = report.get("dropped", 0)
    if dropped:
        line += f", DROPPED {dropped}"
    if report.get("outcome"):
        line += f", {report['outcome']}"
    return line


def episodes(root: Optional[str] = None) -> List[str]:
    """Every episode directory under ``root``, oldest first.

    For the offline export, and for a demo that wants to say how much it has
    collected so far. Sorted by name, which is chronological because the directory
    name starts with the timestamp.
    """
    from pathlib import Path

    base = Path(root or dataset_root())
    if not base.is_dir():
        return []
    return sorted(
        str(entry) for entry in base.iterdir()
        if entry.is_dir() and (entry / "meta.json").exists()
    )
