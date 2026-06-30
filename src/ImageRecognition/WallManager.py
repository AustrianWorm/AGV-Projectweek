import cv2
import numpy as np
from dataclasses import dataclass
import heapq


@dataclass
class WallSegment:
    start: tuple[float, float]   
    end:   tuple[float, float]


_BLUE_LOWER     = np.array([100, 110, 120])   # narrow band around tape color #5199ed
_BLUE_UPPER     = np.array([112, 255, 255])
_WALL_INFLATE_PX = 8   # bigger wall hitbox
_AGV_HITBOX_PX   = 60  # bigger AGV hitbox (collision radius, raw-frame px)


class WallManager:
    def __init__(self, grid_cols: int = 80, grid_rows: int = 50):
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

        self.wall_segments: list[WallSegment] = []
        self.occupancy_grid: np.ndarray = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    def detect_walls(self, image: np.ndarray, border_corners: list[tuple[float, float]] | None = None):
        mask = self._threshold_blue(image)
        self.wall_segments = self._find_segments(mask)
        if border_corners is not None:
            self.wall_segments += self._border_segments(border_corners)
        self.occupancy_grid = self._rasterize(self.wall_segments, image.shape)

    def add_agv_obstacles(self, positions: list[tuple[float, float]], image_shape: tuple[int, int]):
        """Mark other AGVs as round obstacles (bigger hitbox) on the occupancy grid."""
        H, W = image_shape[:2]
        cell_px  = ((W / self.grid_cols) + (H / self.grid_rows)) / 2.0
        radius   = max(1, int(_AGV_HITBOX_PX / cell_px))
        for pos in positions:
            cell = _px_to_cell(pos, W, H, self.grid_cols, self.grid_rows)
            cv2.circle(self.occupancy_grid, cell, radius, color=1, thickness=-1)

    def find_path(
        self,
        start_px: tuple[float, float],
        goal_px:  tuple[float, float],
        image_shape: tuple[int, int],
    ) -> list[tuple[float, float]]:
        H, W       = image_shape
        start_cell = _px_to_cell(start_px, W, H, self.grid_cols, self.grid_rows)
        goal_cell  = _px_to_cell(goal_px,  W, H, self.grid_cols, self.grid_rows)
        cell_path  = _astar(self.occupancy_grid, start_cell, goal_cell)
        return [_cell_to_px(c, W, H, self.grid_cols, self.grid_rows) for c in cell_path]

    def _threshold_blue(self, image: np.ndarray) -> np.ndarray:
        hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _BLUE_LOWER, _BLUE_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return mask

    def _find_segments(self, mask: np.ndarray) -> list[WallSegment]:
        """One wall segment per straight piece of tape (e.g. a triangle yields 3 segments)."""
        skeleton = _skeletonize(mask)
        lines = cv2.HoughLinesP(skeleton, 1, np.pi / 180,
                                 threshold=20, minLineLength=15, maxLineGap=8)
        if lines is None:
            return []
        return [
            WallSegment(start=(float(x1), float(y1)), end=(float(x2), float(y2)))
            for x1, y1, x2, y2 in lines.reshape(-1, 4)
        ]

    def _border_segments(self, corners: list[tuple[float, float]]) -> list[WallSegment]:
        """Outer field boundary as a quad connecting the 4 corner-marker centers (TL,TR,BR,BL)."""
        return [
            WallSegment(start=corners[i], end=corners[(i + 1) % 4])
            for i in range(4)
        ]

    def _rasterize(self, segments: list[WallSegment], image_shape: tuple) -> np.ndarray:
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        H, W = image_shape[:2]

        for seg in segments:
            p1 = _px_to_cell(seg.start, W, H, self.grid_cols, self.grid_rows)
            p2 = _px_to_cell(seg.end,   W, H, self.grid_cols, self.grid_rows)
            cv2.line(grid, p1, p2, color=1, thickness=2)

        kernel = np.ones((_WALL_INFLATE_PX * 2 + 1,) * 2, dtype=np.uint8)
        grid   = cv2.dilate(grid, kernel, iterations=1)

        grid[0, :]  = 1
        grid[-1, :] = 1
        grid[:, 0]  = 1
        grid[:, -1] = 1

        return grid


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Thin tape blobs to 1px centerlines so Hough finds one clean line per edge."""
    try:
        return cv2.ximgproc.thinning(mask)
    except (AttributeError, cv2.error):
        return mask


def _px_to_cell(px, W, H, cols, rows) -> tuple[int, int]:
    col = int(np.clip(px[0] / W * cols, 0, cols - 1))
    row = int(np.clip(px[1] / H * rows, 0, rows - 1))
    return (col, row)


def _cell_to_px(cell, W, H, cols, rows) -> tuple[float, float]:
    return ((cell[0] + 0.5) / cols * W, (cell[1] + 0.5) / rows * H)


def _astar(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = grid.shape

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    neighbours = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    open_heap  = [(h(start, goal), 0, start)]
    came_from: dict = {}
    g_score        = {start: 0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        if g > g_score.get(current, float("inf")):
            continue
        for dc, dr in neighbours:
            nb = (current[0] + dc, current[1] + dr)
            if not (0 <= nb[0] < cols and 0 <= nb[1] < rows):
                continue
            if grid[nb[1], nb[0]]:
                continue
            ng = g + (1.414 if dc and dr else 1.0)
            if ng < g_score.get(nb, float("inf")):
                g_score[nb]   = ng
                came_from[nb] = current
                heapq.heappush(open_heap, (ng + h(nb, goal), ng, nb))

    return []