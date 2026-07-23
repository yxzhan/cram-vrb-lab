"""Robot-agnostic ROS 2 message building and quaternion math for sim nodes.

Free functions only; importable without a SimulationApp (needs the sourced ROS
environment for the message types).
"""

import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image


# --- tiny quaternion helpers (scalar-last x, y, z, w), no external deps ---
def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def qrot(q, v):
    u = q[:3]
    s = q[3]
    return 2 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2 * s * np.cross(u, v)


def as_np(x):
    if hasattr(x, "numpy"):  # warp array or torch CPU tensor
        try:
            return x.numpy()
        except Exception:
            return x.detach().cpu().numpy()
    return np.asarray(x)


def image_msg(rgb, stamp, frame_id):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = rgb.shape[0]
    msg.width = rgb.shape[1]
    msg.encoding = "rgb8"
    msg.step = rgb.shape[1] * 3
    msg.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    return msg


def depth_msg(depth, stamp, frame_id):
    depth = as_np(depth).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    # REP 118: no-data pixels in a 32FC1 depth image are NaN. Isaac returns
    # +inf for rays that hit nothing (beyond the far clip); mark them invalid
    # so depth_image_proc / RViz DepthCloud skip them instead of projecting
    # points at infinity.
    depth = np.where(np.isfinite(depth), depth, np.nan)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = int(depth.shape[0]), int(depth.shape[1])
    msg.encoding = "32FC1"
    msg.step = msg.width * 4
    msg.data = np.ascontiguousarray(depth, dtype=np.float32).tobytes()
    return msg


def build_camera_info(camera, width, height, frame_id):
    """CameraInfo (pinhole intrinsics) for an Isaac camera, computed once.

    Needed by RViz's DepthCloud / depth_image_proc to turn the depth image
    into a point cloud. fx/fy come from the USD focal length and aperture.
    """
    focal = camera.get_focal_length()
    fx = width * focal / camera.get_horizontal_aperture()
    fy = height * focal / camera.get_vertical_aperture()
    cx, cy = width / 2.0, height / 2.0
    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width, info.height = width, height
    info.distortion_model = "plumb_bob"
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def make_tf(stamp, parent, child, p, q):
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = float(p[0])
    t.transform.translation.y = float(p[1])
    t.transform.translation.z = float(p[2])
    t.transform.rotation.x = float(q[0])
    t.transform.rotation.y = float(q[1])
    t.transform.rotation.z = float(q[2])
    t.transform.rotation.w = float(q[3])
    return t
