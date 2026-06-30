import argparse
import json

import cv2
import numpy as np
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

def _load_config(path: str = "config.json") -> dict:
    """Read the team's ArUco-ID configuration file."""
    with open(path) as f:
        return json.load(f)


def _own_and_goal_ids(config: dict) -> tuple[int, int]:
    """Own AGV marker ID and goal marker ID, supporting multiple config formats."""
    if "ArUcoCodes" in config:
        codes = config["ArUcoCodes"]
        return codes.get("ArUcoSelf", 9), codes.get("goal", 10)
    elif "agv" in config:
        agv_config = config["agv"]
        own_id = agv_config.get("ArUcoSelf", config.get("ArUcoSelf", 9))
        goal_id = agv_config.get("goal", config.get("goal", 10))
        return own_id, goal_id
    return 9, 10


def debugCVControl(url: str, config_path: str = "config.json"):
    """Run the full image-recognition pipeline against the live stream and show it."""
    from ImageRecognition.ImageAnalyzationController import ImageAnalyzationController
    from ImageRecognition.WallManager import WallManager

    own_id, goal_id = _own_and_goal_ids(_load_config(config_path))

    cam      = OpenCVControl(stream_url=url)
    aruco    = Arucomanager(agv_marker_id=own_id)
    walls    = WallManager()
    pipeline = ImageAnalyzationController(cam, aruco, walls, goal_marker_id=goal_id)
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


# ── Main Runnable block ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Setup argparse to dynamically parse config location if required
    parser = argparse.ArgumentParser(description="Run OpenCV Control Pipeline using config specs.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json", 
        help="Path to the configuration file (default: config.json)"
    )
    args = parser.parse_args()

    # 2. Open and load configuration as instructed
    with open(args.config) as f:
        config = json.load(f)

    # 3. Safely extract stream URL from config["agv"]["agv_url"]
    stream_url = config["agv"]["agv_url"]
    print(f"Connecting to stream URL from config: {stream_url}")

    # 4. Trigger the main visualization pipeline entry point
    debugCVControl(url=stream_url, config_path=args.config)