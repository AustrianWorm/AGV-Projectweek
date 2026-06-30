import cv2
import numpy as np

from ImageRecognition.OpenCvControl import OpenCVControl
from ImageRecognition.Arucomanager  import Arucomanager
from ImageRecognition.WallManager   import WallManager

GOAL_MARKER_ID = 10  # ArUco marker that marks the maze exit / goal

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

        self.raw_frame: np.ndarray | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def start_image_analysis(self) -> bool:
        """
        Run one full analysis cycle, entirely in raw (unrectified) camera space:
        grab → detect markers → detect walls/border → mark AGV hitboxes.
        Returns True if a frame was successfully grabbed.
        """
        frame = self._camera.get_frame()
        if frame is None:
            return False
        self.raw_frame = frame

        self._aruco.start_aruco_detection(frame)

        border_corners = self._field_border_corners()
        self._walls.detect_walls(frame, border_corners)
        self._walls.add_agv_obstacles(self._agv_positions(), frame.shape)
        return True

    def stop_image_analysis(self):
        """Clear analysis state."""
        self._aruco.stop_aruco_detection()
        self.raw_frame = None

    def draw_debug(self) -> np.ndarray:
        """Return the raw frame annotated with markers, walls/border, and AGV hitboxes."""
        vis = self.raw_frame.copy() if self.raw_frame is not None else np.zeros((480, 640, 3), np.uint8)
        self._draw_markers(vis)
        self._draw_walls(vis)
        self._draw_agv_hitboxes(vis)
        self._draw_goal(vis)
        return vis

    # ── Internal ──────────────────────────────────────────────────────────────

    def _field_border_corners(self) -> list[tuple[float, float]] | None:
        """Centers of the four corner markers, in raw-frame pixels, if all are visible."""
        if not self._aruco.is_field_locked():
            return None
        return [self._aruco.markers[cid].center for cid in self._aruco.corner_ids]

    def _agv_positions(self) -> list[tuple[float, float]]:
        """Raw-frame positions of every AGV marker (anything that isn't a corner or the goal)."""
        return [
            pose.center
            for marker_id, pose in self._aruco.markers.items()
            if marker_id not in self._aruco.corner_ids and marker_id != GOAL_MARKER_ID
        ]

    def _draw_markers(self, vis: np.ndarray):
        """Draw marker outlines, IDs, and Team details in-place."""
        for marker_id, pose in self._aruco.markers.items():
            color = (0, 200, 0) if marker_id != self._aruco.agv_marker_id else (0, 100, 255)
            pts   = pose.corners.astype(int)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

            label = f"ID:{marker_id}"
            if marker_id in TEAM_MAPPING:
                label += f" ({TEAM_MAPPING[marker_id]})"

            cv2.putText(vis, label,
                        (pts[0][0], pts[0][1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_walls(self, vis: np.ndarray):
        """Draw detected tape walls and field border as bold blue lines, in-place."""
        for seg in self._walls.wall_segments:
            cv2.line(vis,
                     (int(seg.start[0]), int(seg.start[1])),
                     (int(seg.end[0]),   int(seg.end[1])),
                     (255, 180, 0), 5)

    def _draw_agv_hitboxes(self, vis: np.ndarray):
        """Draw a yellow hitbox circle around every AGV marker (not corners, not the goal), in-place."""
        for pos in self._agv_positions():
            cx, cy = int(pos[0]), int(pos[1])
            cv2.circle(vis, (cx, cy), 60, (0, 220, 255), 3)

    def _draw_goal(self, vis: np.ndarray):
        """Highlight the goal marker in green, in-place."""
        goal = self._aruco.markers.get(GOAL_MARKER_ID)
        if goal is None:
            return
        gx, gy = int(goal.center[0]), int(goal.center[1])
        cv2.circle(vis, (gx, gy), 16, (0, 220, 0), 3)
        cv2.putText(vis, "GOAL", (gx - 18, gy - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)