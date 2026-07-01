import argparse
import json
import cv2

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

def _load_config(path: str = "config.json") -> dict:
    """Read the team's shared configuration file."""
    with open(path) as f:
        return json.load(f)


def debugCVControl(url: str, config_path: str = "config.json"):
    """Run the full image-recognition pipeline initialized from config.json data."""
    from ImageRecognition.ImageAnalyzationController import ImageAnalyzationController
    from ImageRecognition.Arucomanager import Arucomanager
    from ImageRecognition.WallManager import WallManager

    # Load configuration file object mapping
    config = _load_config(config_path)

    cam      = OpenCVControl(stream_url=url)
    aruco    = Arucomanager(config=config)
    walls    = WallManager(config=config)
    pipeline = ImageAnalyzationController(cam, aruco, walls, config=config)
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


# ── CLI Execution Entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start OpenCvControl Pipeline with central configuration.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json", 
        help="Path to setup config file (default: config.json)"
    )
    args = parser.parse_args()

    # Load file
    with open(args.config) as f:
        config = json.load(f)

    # Grab agv stream endpoint
    stream_url = config["agv"]["agv_url"]
    
    print(f"Loaded config from: {args.config}")
    print(f"Connecting to stream target: {stream_url}")
    
    # Run pipeline loop
    debugCVControl(url=stream_url, config_path=args.config)