import cv2
import numpy as np

from OpenCvControl import OpenCVControl
from Arucomanager  import Arucomanager
from WallManager   import WallManager

# Lookup dictionary for ArUco marker IDs mapped to Team Numbers and Member Names
TEAM_MAPPING = {
    100: "Team 22: Wimmer Julian, Schacherbauer Moritz, Kohlmayer David",
    97: "Team 21: Sigmund Lukas, Falch David",
    33: "Team 20: Wimmer Jakob, Börner Leon",
    69: "Team 19: Vorreiter Tobias, Berer Moritz",
    16: "Team 16: Kolev Victor, Vasic Dorde",
    42: "Team 15: Bandat Jonathan, Pavalacs Barna",
    10: "Team 13: Hengstberger Simon, Herejk Simon",
    9: "Team 12: Mühlbacher Paul, Enzinger Mark",
    21: "Team 11: Zadny Raphael, Klinger Fabian",
    73: "Team 9: Medland Ben, Klepp Bastian",
    24: "Team 8: Barbu Stefan, Wührer-Silberer Simon Hermann",
    5: "Team 5: Lauss Valentin, Neuhauser Lukas",
    44: "Team 3: Wojakowski Jakub, Wurmhöringer David",
    101: "Team 2: Bonitz Timo, Kastner Andreas",
    67: "Team 1: Wolfsegger Johannes, Wohlfarter Florian"
}


class ImageAnalyzationController:
    def __init__(
        self,
        camera: OpenCVControl,
        aruco:  Arucomanager,
        walls:  WallManager,
    ):
        self._camera = camera
        self._aruco  = aruco
        self._walls  = walls

        self.raw_frame:  np.ndarray | None = None
        self.rect_frame: np.ndarray | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def start_image_analysis(self) -> bool:
        """
        Run one full analysis cycle: grab → detect markers → rectify → detect walls.
        Returns True if a rectified frame was successfully produced.
        """
        frame = self._camera.get_frame()
        if frame is None:
            return False
        self.raw_frame = frame

        self._aruco.start_aruco_detection(frame)

        if not self._aruco.is_field_locked():
            return False

        self.rect_frame = self._rectify(frame)
        self._walls.detect_walls(self.rect_frame)
        return True

    def stop_image_analysis(self):
        """Clear analysis state."""
        self._aruco.stop_aruco_detection()
        self.raw_frame  = None
        self.rect_frame = None

    def draw_debug(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (annotated_raw, annotated_rect) for display."""
        raw_vis  = self._draw_markers(
            self.raw_frame.copy() if self.raw_frame is not None
            else np.zeros((480, 640, 3), np.uint8)
        )
        rect_vis = self._draw_rect_overlays(
            self.rect_frame.copy() if self.rect_frame is not None
            else np.zeros((self._aruco.rect_height, self._aruco.rect_width, 3), np.uint8)
        )
        return raw_vis, rect_vis

    # ── Internal ──────────────────────────────────────────────────────────────

    def _rectify(self, frame: np.ndarray) -> np.ndarray:
        """Warp raw frame into a clean top-down view using the ArUco warp matrix."""
        return cv2.warpPerspective(
            frame,
            self._aruco.warp_matrix,
            (self._aruco.rect_width, self._aruco.rect_height),
        )

    def _draw_markers(self, vis: np.ndarray) -> np.ndarray:
        """Draw marker outlines, IDs, and Team details onto a raw frame copy."""
        for marker_id, pose in self._aruco.markers.items():
            color = (0, 200, 0) if marker_id != self._aruco.agv_marker_id else (0, 100, 255)
            pts   = pose.corners.astype(int)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
            
            # Construct label string with appended team name/members if existing
            label = f"ID:{marker_id}"
            if marker_id in TEAM_MAPPING:
                label += f" ({TEAM_MAPPING[marker_id]})"

            cv2.putText(vis, label,
                        (pts[0][0], pts[0][1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return vis

    def _draw_rect_overlays(self, vis: np.ndarray) -> np.ndarray:
        """Draw walls, AGV pose, and Team details onto a rectified frame copy."""
        # Walls (cyan)
        for seg in self._walls.wall_segments:
            cv2.line(vis,
                     (int(seg.start[0]), int(seg.start[1])),
                     (int(seg.end[0]),   int(seg.end[1])),
                     (255, 200, 0), 2)

        # AGV heading arrow (orange)
        agv = self._aruco.get_agv()
        if agv is not None:
            cx, cy = int(agv.center[0]), int(agv.center[1])
            ex = int(cx + 40 * np.cos(np.radians(agv.heading)))
            ey = int(cy - 40 * np.sin(np.radians(agv.heading)))
            cv2.circle(vis, (cx, cy), 6, (0, 100, 255), -1)
            cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 60, 255), 2, tipLength=0.3)
            
            # Label on the top-down perspective screen
            label = f"ID:{agv.id}"
            if agv.id in TEAM_MAPPING:
                label += f" ({TEAM_MAPPING[agv.id]})"
            cv2.putText(vis, label, (cx - 15, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

        return vis