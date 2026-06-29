@dataclass  
class WallSegment:
    start: tuple[float, float]   # (x, y) in rectified pixel-space
    end:   tuple[float, float]


class WallManager:
    def __init__(self):
        self.wall_segments: list[WallSegment] = []

    def detect_walls(self, image):
        # Placeholder for wall detection logic
        # This method should analyze the image and populate self.wall_segments
        return