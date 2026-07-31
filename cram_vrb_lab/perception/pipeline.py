"""Run robokudo's geometric RGB-D pipeline against the Stretch's head camera.

The apartment is described twice and the two descriptions disagree: the objects
standing on the kitchen counter (``SM_MilkBox``, ``SM_CerealBox``, ``SM_SmallBowl``,
``SM_Cup``) exist in ``apartmentICRA.usda``, which Isaac renders, and **not** in
``apartment.urdf``, which giskard and the digital twin are built from. So the camera
can see them and the twin cannot. This module closes that gap by looking, rather than
by reading ground truth out of the simulator the way
:class:`cram_vrb_lab.scenes.props.twin_props.CubePoseSensor` does for the
pick-and-place demos.

What comes back is **unlabelled**: plane fit plus clustering gives "there are N things
standing on that surface, here is each one's pose and extent". Naming them would need
a classifier, and ``torch``/``ultralytics`` are not installed in this venv, so
robokudo's learned annotators cannot run.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------------
# Keep robokudo's ROS visualizer -- and therefore cv_bridge -- out of this process.
#
# ``robokudo.utils.tree_execution`` imports ``robokudo.garden``, which imports
# ``robokudo.vis.visualizer_manager`` -> ``robokudo.vis.ros_visualizer`` ->
# ``cv_bridge``, all at *module* level. That chain runs unconditionally, even though
# :func:`detect` passes ``include_gui=False`` and ``garden.grow_tree`` therefore never
# instantiates ``VisualizationManager``: we pay for the import of a visualizer we
# never build.
#
# The system ``cv_bridge`` is compiled against NumPy 1.x and this venv has NumPy 2.x,
# so ``cv_bridge/__init__.py``'s ``from cv_bridge.boost.cv_bridge_boost import ...``
# trips NumPy's compatibility shim. Note what that shim does, because it explains why
# ``try``/``except`` is not the tool here: ``numpy/core/_multiarray_umath.py`` writes
# its "A module that was compiled using NumPy 1.x" page *plus a formatted traceback*
# straight to ``sys.stderr`` and only then raises ``ImportError`` -- and cv_bridge
# catches that ImportError itself (``except ImportError: pass``). So the alarming
# output in the notebook is a **print, not an exception**; there is nothing to catch,
# and wrapping the robokudo imports in ``try``/``except`` cannot suppress it.
# --------------------------------------------------------------------------------

_ROS_VISUALIZER_MODULE = "robokudo.vis.ros_visualizer"


def _disable_ros_visualizer() -> None:
    """Pre-register a stub for :mod:`robokudo.vis.ros_visualizer` in ``sys.modules``.

    ``visualizer_manager`` only does ``from robokudo.vis.ros_visualizer import
    SharedROSVisualizer, AllAnnotatorROSVisualizer`` and lists the two classes in
    ``self.visualizer_types``; they are instantiated solely by
    ``create_visualizers_for_pipeline``, which no ``include_gui=False`` run reaches.
    Satisfying those two names is therefore enough, and the real module -- the only
    importer of ``cv_bridge`` in all of robokudo -- never loads.

    The stubs raise if anything ever does try to build them, so a future
    ``include_gui=True`` fails loudly here rather than mystifyingly inside cv_bridge.
    """
    if _ROS_VISUALIZER_MODULE in sys.modules:
        return

    class _DisabledROSVisualizer:
        """Placeholder for a robokudo ROS visualizer that this venv cannot run."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                f"{type(self).__name__} is disabled by "
                "cram_vrb_lab.perception.pipeline: it needs cv_bridge, which is "
                "compiled against NumPy 1.x while this venv runs NumPy "
                f"{np.__version__}. The geometric pipeline does not need any "
                "visualizer -- detect() ticks with include_gui=False."
            )

    class SharedROSVisualizer(_DisabledROSVisualizer):
        pass

    class AllAnnotatorROSVisualizer(_DisabledROSVisualizer):
        pass

    stub = types.ModuleType(_ROS_VISUALIZER_MODULE)
    stub.__doc__ = "Stub installed by cram_vrb_lab.perception.pipeline; see that module."
    stub.SharedROSVisualizer = SharedROSVisualizer
    stub.AllAnnotatorROSVisualizer = AllAnnotatorROSVisualizer
    stub._cram_vrb_lab_stub = True
    sys.modules[_ROS_VISUALIZER_MODULE] = stub


_disable_ros_visualizer()

# Import order matters. ``robokudo.annotators.outputs`` imports back into
# ``robokudo.pipeline``, so pulling it in as the first robokudo submodule raises
# "cannot import name 'AnnotatorOutputPerPipelineMap' ... partially initialized
# module". The package modules below have to come first; this is the same order
# ``test/robokudo_test/test_full_ae_execution.py`` uses.
import rclpy
import robokudo.cas
import robokudo.pipeline
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.tree_execution
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outputs import ClearAnnotatorOutputs
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.cas import CASViews
from robokudo.descriptors.camera_configs.registry import CameraConfigRegistry
from robokudo.io.camera_interface import KinectCameraInterface

from cram_vrb_lab.robots.stretch.joints import (
    CAMERA_FRAME_ID,
    DEPTH_IMAGE_TOPIC,
    RGB_IMAGE_TOPIC,
    RGB_INFO_TOPIC,
)

PIPELINE_NODE_NAME = "robokudo_pipeline"

CLUSTER_TUNING = {
    # robokudo's defaults were tuned for a Kinect about a metre from a tabletop. The
    # Stretch looks at this counter from ~2.4 m, where the milk box covers roughly
    # 12x38 px -- a few hundred points. The stock min_cluster_count of 1000 rejects
    # every object in the scene, and the pipeline then "succeeds" having found
    # nothing at all.
    "min_cluster_count": 100,
    "dbscan_min_cluster_count": 15,
    "min_on_plane_point_count": 20,
}

CAMERA_CROP = {
    # Camera optical frame: +x right, +y down, +z forward. Keeps the counter and
    # drops the floor and the far wall, either of which the plane fit would
    # otherwise happily choose over the worktop.
    "min_z": 0.3,
    "max_z": 3.0,
    "min_x": -1.0,
    "max_x": 1.0,
    "min_y": -0.6,
    "max_y": 0.8,
}


@dataclass
class Detection:
    """One cluster robokudo found, in the camera's optical frame.

    Deliberately a plain dataclass rather than robokudo's ``ObjectHypothesis``: the
    twin side should not have to know robokudo's annotation types, and the pose still
    needs transforming out of the camera frame -- see
    :mod:`cram_vrb_lab.perception.twin_objects`.
    """

    position: np.ndarray
    """(3,) centre of the cluster, in ``camera_color_optical_frame`` [m]."""

    orientation: np.ndarray
    """(4,) quaternion in ``(x, y, z, w)`` order, robokudo's convention."""

    extents: np.ndarray
    """(3,) side lengths of the oriented bounding box [m]."""

    @property
    def volume(self) -> float:
        return float(np.prod(self.extents))


def _depth_metres_to_millimetres(depth):
    """Isaac publishes ``32FC1`` metres; robokudo's cloud builder wants millimetres.

    ``ImagePreprocessorAnnotator`` states "Depth values are expected to be in
    millimeters" and calls ``o3d.geometry.RGBDImage.create_from_color_and_depth``
    *without* a ``depth_scale``, so Open3D's default of 1000 applies. Feed it metres
    and every point lands within a few millimetres of the origin -- and the plane fit
    still reports SUCCESS, so the mistake is silent.

    Invalid pixels arrive as ``NaN`` (REP 118, see ``sim/ros_utils.py::depth_msg``)
    and become 0, which is what Open3D treats as "no measurement".
    """
    metres = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(metres) & (metres > 0.0)
    millimetres = np.where(valid, metres * 1000.0, 0.0)
    return np.clip(millimetres, 0.0, 65535.0).astype(np.uint16)


class StretchHeadCameraInterface(KinectCameraInterface):
    """robokudo's RGB-D-over-ROS reader, with Isaac's depth units corrected.

    Overrides ``set_data`` rather than ``callback``: the two share a non-reentrant
    lock, and converting after ``callback`` would leave a window where the interface
    advertises new data that is still in metres.
    """

    def set_data(self, cas) -> None:
        had_new_data = self.has_new_data()
        super().set_data(cas)
        if not had_new_data:
            return
        cas.set(
            CASViews.DEPTH_IMAGE,
            _depth_metres_to_millimetres(cas.get(CASViews.DEPTH_IMAGE)),
        )


def camera_descriptor():
    """A collection-reader descriptor pointed at the Stretch's head camera.

    Built directly rather than through ``CollectionReaderDescriptorFactory`` so the
    depth-correcting interface above can be injected; the factory hard-codes one
    interface class per camera name.

    Topic names come from :mod:`cram_vrb_lab.robots.stretch.joints` rather than being
    retyped, so they cannot drift from what the sim publishes. ``*_hints="raw"``
    because Isaac publishes uncompressed ``rgb8``/``32FC1``;
    ``color2depth_ratio=(1, 1)`` and ``hi_res_mode=False`` because both images come
    from the *same* Isaac camera at one resolution in one frame, so they are already
    pixel-aligned and share a single ``CameraInfo``.

    ``lookup_viewpoint=True`` is required, not cosmetic: ``ClusterPoseBBAnnotator``
    returns ``FAILURE`` with "Couldn't find camera viewpoint in the CAS" if the
    camera-to-world transform was never put there.
    """
    config = CameraConfigRegistry.create_config(
        "kinect",
        topic_color=RGB_IMAGE_TOPIC,
        topic_depth=DEPTH_IMAGE_TOPIC,
        topic_camera_info=RGB_INFO_TOPIC,
        color_hints="raw",
        depth_hints="raw",
        color2depth_ratio=(1.0, 1.0),
        hi_res_mode=False,
        tf_from=CAMERA_FRAME_ID,
        tf_to="map",
        lookup_viewpoint=True,
    )
    return CollectionReaderAnnotator.Descriptor(
        camera_config=config,
        camera_interface=StretchHeadCameraInterface(config),
    )


def build_pipeline(descriptor, crop=None, tuning=None):
    """The geometric analysis engine: crop, fit the dominant plane, cluster what
    stands on it, fit an oriented box to each cluster.

    :param descriptor: from :func:`camera_descriptor`. Pass the *same* one across
        runs -- constructing it spins up a camera node and a subscription set.

    Build a fresh pipeline per run; py_trees keeps blackboard state on the object.
    """
    crop_config = PointcloudCropAnnotator.Descriptor()
    for name, value in {**CAMERA_CROP, **(crop or {})}.items():
        setattr(crop_config.parameters, name, value)
    # Leave relative_to_world False. Besides being wrong for a camera-frame crop,
    # PointcloudCropAnnotator.update puts its `return Status.FAILURE` inside the
    # relative_to_world branch but outside the try/except, so setting it True fails
    # unconditionally.
    crop_config.parameters.relative_to_world = False

    cluster_config = PointCloudClusterExtractor.Descriptor()
    for name, value in {**CLUSTER_TUNING, **(tuning or {})}.items():
        setattr(cluster_config.parameters, name, value)

    pipeline = robokudo.pipeline.Pipeline("StretchHeadCameraPipeline")
    pipeline.add_children(
        [
            ClearAnnotatorOutputs(),
            CollectionReaderAnnotator(descriptor=descriptor),
            ImagePreprocessorAnnotator("ImagePreprocessor"),
            PointcloudCropAnnotator(descriptor=crop_config),
            PlaneAnnotator(),
            PointCloudClusterExtractor(descriptor=cluster_config),
            ClusterPoseBBAnnotator(),
        ]
    )
    return pipeline


def make_pipeline_node():
    """A dedicated rclpy node for ticking the pipeline.

    .. warning::
       Never pass the notebook's own node here.
       ``robokudo.utils.tree_execution._tick_tree_until`` builds its own
       ``MultiThreadedExecutor`` and calls ``executor.add_node(node)``, which in rclpy
       *detaches the node from the executor already spinning it*, then
       ``remove_node``s it in a ``finally``. The notebook's node would come back
       attached to nothing, and ``WorldSynchronizer`` plus every giskard action client
       on it would silently stop working for the rest of the session.
    """
    return rclpy.create_node(PIPELINE_NODE_NAME)


def detect(pipeline_node, descriptor, pipeline=None, max_iterations=120, tick_rate=10):
    """Tick the pipeline until one RGB-D frame has been processed; return the
    detections, still in the camera frame.

    :param pipeline_node: from :func:`make_pipeline_node` -- *not* the notebook's node.
    :param max_iterations: tick budget. If the sim was started without
        ``camera="both"`` no frame ever arrives, and this is what stops it hanging.

    Ticks a bounded number of times rather than calling robokudo's ``run_ae``, which
    loops forever and is meant for a standalone process.
    """
    pipeline = pipeline if pipeline is not None else build_pipeline(descriptor)

    status = robokudo.utils.tree_execution.run_tree_once(
        pipeline,
        pipeline_node,
        include_gui=False,
        max_iterations=max_iterations,
        tick_rate=tick_rate,
    )
    if str(status) != "Status.SUCCESS":
        raise TimeoutError(
            f"robokudo did not process a frame (status {status}) in {max_iterations} "
            f"ticks. Check that {RGB_IMAGE_TOPIC} and {DEPTH_IMAGE_TOPIC} are "
            f"publishing -- the sim must be started with camera='both' -- and that "
            f"tf can resolve map -> {CAMERA_FRAME_ID}."
        )
    return detections_from_cas(pipeline.cas)


def detections_from_cas(cas):
    """Pull the pose and bounding box off each object hypothesis in a CAS."""
    detections = []
    for hypothesis in cas.filter_annotations_by_type(
        robokudo.types.scene.ObjectHypothesis
    ):
        boxes = [
            annotation
            for annotation in hypothesis.annotations
            if isinstance(annotation, robokudo.types.annotation.BoundingBox3DAnnotation)
        ]
        if not boxes:
            # A cluster the pose/bb annotator could not fit a box to; nothing useful
            # to put in the twin.
            continue
        box = boxes[0]
        detections.append(
            Detection(
                position=np.asarray(box.pose.translation, dtype=float),
                orientation=np.asarray(box.pose.rotation, dtype=float),
                extents=np.array(
                    [box.x_length, box.y_length, box.z_length], dtype=float
                ),
            )
        )
    return detections
