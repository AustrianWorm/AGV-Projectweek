import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class MarkerPose:
    id: int
    center: tuple[float, float]   # (x, y) in rectified pixels
    heading: float                 # degrees
    corners: np.ndarray           # (4,2) raw corners, kept for re-warping


# ArUco marker ID of this team's AGV — set before use
# Corner marker IDs for the four field corners (TL, TR, BR, BL)
CORNER_IDS = [1, 2, 3, 4]   # update on the day
AGV_MARKER_ID = 9           # Self: ArUcoMarker 44

ARUCO_DICT   = cv2.aruco.DICT_4X4_100
RECT_WIDTH   = 800
RECT_HEIGHT  = 600


class Arucomanager:
    def __init__(self, agv_marker_id: int = AGV_MARKER_ID, corner_ids: list[int] = None):
        self.agv_marker_id = agv_marker_id
        self.corner_ids    = corner_ids or CORNER_IDS
        self.rect_width    = RECT_WIDTH
        self.rect_height   = RECT_HEIGHT

        self._detector  = _make_detector()
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

def _make_detector() -> cv2.aruco.ArucoDetector:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params     = cv2.aruco.DetectorParameters()
    # Wider tolerance: catches markers at odd angles, blur, or in low contrast
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
        # heading: angle of the top edge (corner 0 → corner 1)
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
    """Perspective transform from four corner markers → top-down rectangle.
    corner_ids order: [TL, TR, BR, BL]
    """
    src = np.array(
        [markers[cid].center for cid in corner_ids],
        dtype=np.float32,
    )
    dst = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)