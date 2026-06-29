import cv2
import numpy as np
from dataclasses import dataclass
import heapq


@dataclass
class WallSegment:
    start: tuple[float, float]   
    end:   tuple[float, float]


_BLUE_LOWER     = np.array([100,  80,  50])
_BLUE_UPPER     = np.array([130, 255, 255])
_WALL_INFLATE_PX = 2


class WallManager:
    def __init__(self, grid_cols: int = 80, grid_rows: int = 50):
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

        self.wall_segments: list[WallSegment] = []
        self.occupancy_grid: np.ndarray = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    def detect_walls(self, image: np.ndarray):
        mask = self._threshold_blue(image)
        self.wall_segments  = self._find_segments(mask)
        self.occupancy_grid = self._rasterize(self.wall_segments, image.shape)

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
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segments = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:
                continue
            (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
            angle  = np.radians(angle)
            half   = max(w, h) / 2.0
            offset = np.pi / 2 if w < h else 0
            dx = np.cos(angle + offset) * half
            dy = np.sin(angle + offset) * half
            segments.append(WallSegment(start=(cx - dx, cy - dy), end=(cx + dx, cy + dy)))
        return segments

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