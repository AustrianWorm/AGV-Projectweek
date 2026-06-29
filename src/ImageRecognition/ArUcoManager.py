@dataclass
class MarkerPose:
    id: int
    center: tuple[float, float]   # (x, y) in rectified pixels
    heading: float                 # degrees
    corners: np.ndarray           # (4,2) raw corners, kept for re-warping


// Self: ArUcoMarker 44

class ArUcoManager:
    def __init__(self):
        self.aurco_manager = ArUcoManager()

    def detected_markers(self):
        return []