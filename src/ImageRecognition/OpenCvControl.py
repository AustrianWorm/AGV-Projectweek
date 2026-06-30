import cv2
from ImageRecognition.Arucomanager import Arucomanager

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


class OpenCVControl:
    def __init__(self, stream_url: str):
        self.stream_url   = stream_url
        self._cap: cv2.VideoCapture | None = None
        self._viewing     = False
        self._window_name = "Webstream Viewer"

    def connect(self):
        self._cap = cv2.VideoCapture(self.stream_url)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(f"Could not open stream at {self.stream_url}.")

    def disconnect(self):
        self.stop_view()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frame(self) -> "np.ndarray | None":
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def start_view(self):
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Not connected. Call connect() first.")
        self._viewing = True
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        while self._viewing:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._cap.release()
                self._cap = cv2.VideoCapture(self.stream_url)
                if not self._cap.isOpened():
                    self._viewing = False
                    break
                continue
            cv2.imshow(self._window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.stop_view()

    def stop_view(self):
        self._viewing = False
        cv2.destroyAllWindows()


# ── Debug entry point ─────────────────────────────────────────────────────────

def debugCVControl(url: str):
    """Run the full image-recognition pipeline against the live stream and show it."""
    from ImageRecognition.ImageAnalyzationController import ImageAnalyzationController
    from ImageRecognition.WallManager import WallManager

    cam      = OpenCVControl(stream_url=url)
    aruco    = Arucomanager()
    walls    = WallManager()
    pipeline = ImageAnalyzationController(cam, aruco, walls)
    cam.connect()

    cv2.namedWindow("Raw", cv2.WINDOW_NORMAL)

    while True:
        pipeline.start_image_analysis()
        vis = pipeline.draw_debug()

        cv2.imshow("Raw", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pipeline.stop_image_analysis()
    cam.disconnect()