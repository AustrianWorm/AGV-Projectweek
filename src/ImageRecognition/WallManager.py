import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class WallSegment:
    start: tuple[float, float]   # (x, y) in rectified pixel-space
    end:   tuple[float, float]


# HSV range for blue tape — tune with a colour picker if lighting changes
_BLUE_LOWER = np.array([100,  80,  50])
_BLUE_UPPER = np.array([130, 255, 255])


class WallManager:
    def __init__(self, grid_cols: int = 80, grid_rows: int = 50):
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

        self.wall_segments: list[WallSegment] = []
        # Binary occupancy grid: 0 = free, 1 = wall. Shape (grid_rows, grid_cols).
        self.occupancy_grid: np.ndarray = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    # ── Public ────────────────────────────────────────────────────────────────

    def detect_walls(self, image: np.ndarray):
        """
        Detect blue tape walls in a rectified top-down image.
        Updates self.wall_segments and self.occupancy_grid.
        """
        mask = self._threshold_blue(image)
        self.wall_segments = self._find_segments(mask)
        self.occupancy_grid = self._rasterize(self.wall_segments, image.shape)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _threshold_blue(self, image: np.ndarray) -> np.ndarray:
        """Return a binary mask of blue-tape pixels."""
        hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _BLUE_LOWER, _BLUE_UPPER)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return mask

    def _find_segments(self, mask: np.ndarray) -> list[WallSegment]:
        """Fit a line segment to each contour blob in the mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segments = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:   # skip noise
                continue
            cx, cy, (w, h), angle = (*cv2.minAreaRect(cnt)[0],
                                      cv2.minAreaRect(cnt)[1],
                                      cv2.minAreaRect(cnt)[2])
            cx, cy = cv2.minAreaRect(cnt)[0]
            w, h   = cv2.minAreaRect(cnt)[1]
            angle  = np.radians(cv2.minAreaRect(cnt)[2])
            half   = max(w, h) / 2.0
            offset = np.pi / 2 if w < h else 0
            dx = np.cos(angle + offset) * half
            dy = np.sin(angle + offset) * half
            segments.append(WallSegment(
                start=(cx - dx, cy - dy),
                end  =(cx + dx, cy + dy),
            ))
        return segments

    def _rasterize(self, segments: list[WallSegment], image_shape: tuple) -> np.ndarray:
        """Draw segments onto the occupancy grid. Also marks the outer boundary."""
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        H, W = image_shape[:2]

        def to_cell(x: float, y: float) -> tuple[int, int]:
            col = int(np.clip(x / W * self.grid_cols, 0, self.grid_cols - 1))
            row = int(np.clip(y / H * self.grid_rows, 0, self.grid_rows - 1))
            return col, row   # cv2.line expects (x, y) = (col, row)

        for seg in segments:
            p1 = to_cell(*seg.start)
            p2 = to_cell(*seg.end)
            cv2.line(grid, p1, p2, color=1, thickness=2)

        # Outer boundary is always occupied
        grid[0, :]  = 1
        grid[-1, :] = 1
        grid[:, 0]  = 1
        grid[:, -1] = 1

        return grid