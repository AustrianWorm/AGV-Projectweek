import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class MarkerPose:
    id: int
    center: tuple[float, float]   # (x, y) in rectified pixels
    heading: float                 # degrees
    corners: np.ndarray           # (4,2) raw corners, kept for re-warping

# String-to-CV2 constant dictionary lookup map
ARUCO_DICT_MAPPING = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}

class Arucomanager:
    def __init__(self, config: dict):
        codes_cfg = config.get("ArUcoCodes", {})
        aruco_cfg = config.get("aruco_settings", {})

        self.agv_marker_id = codes_cfg.get("ArUcoSelf", 9)
        self.corner_ids    = codes_cfg.get("corners", [1, 2, 3, 4])
        self.rect_width    = aruco_cfg.get("rect_width", 800)
        self.rect_height   = aruco_cfg.get("rect_height", 600)

        dict_str = aruco_cfg.get("dict", "DICT_4X4_100")
        aruco_dict_const = ARUCO_DICT_MAPPING.get(dict_str, cv2.aruco.DICT_4X4_100)

        self._detector  = _make_detector(aruco_dict_const)
        self.markers:   dict[int, MarkerPose] = {}
        self.warp_matrix: np.ndarray | None   = None

    # ── Public ────────────────────────────────────────────────────────────────

    def start_aruco_detection(self, frame: np.ndarray):
        """Detect all ArUco markers in frame; update warp matrix if field is visible."""
        self.markers = _detect_markers(frame, self._detector)
        if self.is_field_locked():
            self.warp_matrix = _compute_warp(
                self.markers, self.corner_ids,
                self.rect_width, self.rect_height,
            )

    def stop_aruco_detection(self):
        self.markers     = {}
        self.warp_matrix = None

    def is_field_locked(self) -> bool:
        """True when all four corner markers are visible."""
        return all(cid in self.markers for cid in self.corner_ids)

    def get_agv(self) -> MarkerPose | None:
        return self.markers.get(self.agv_marker_id)

    def detected_markers(self) -> list[MarkerPose]:
        return list(self.markers.values())


# ── Module-level helpers ──────────────────────────────────────────────────────

def _make_detector(aruco_dict_const) -> cv2.aruco.ArucoDetector:
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_const)
    params     = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin    = 3
    params.adaptiveThreshWinSizeMax    = 53
    params.adaptiveThreshWinSizeStep   = 4
    params.minMarkerPerimeterRate      = 0.01
    params.polygonalApproxAccuracyRate = 0.05
    params.errorCorrectionRate         = 0.8
    params.cornerRefinementMethod      = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def _detect_markers(frame: np.ndarray, detector: cv2.aruco.ArucoDetector) -> dict[int, MarkerPose]:
    corners_list, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return {}
    result = {}
    for corners, marker_id in zip(corners_list, ids.flatten()):
        pts    = corners[0]                        # shape (4, 2)
        center = pts.mean(axis=0)
        dx      = pts[1][0] - pts[0][0]
        dy      = pts[1][1] - pts[0][1]
        heading = -np.degrees(np.arctan2(dy, dx))
        result[int(marker_id)] = MarkerPose(
            id=int(marker_id),
            center=(float(center[0]), float(center[1])),
            heading=float(heading),
            corners=pts,
        )
    return result


def _compute_warp(
    markers: dict[int, MarkerPose],
    corner_ids: list[int],
    width: int,
    height: int,
) -> np.ndarray:
    """Perspective transform from four corner markers → top-down rectangle."""
    src = np.array(
        [markers[cid].center for cid in corner_ids],
        dtype=np.float32,
    )
    dst = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)