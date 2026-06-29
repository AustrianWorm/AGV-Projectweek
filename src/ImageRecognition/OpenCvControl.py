"""
Webstream Viewer using OpenCV
Connects to: http://10.250.150.224:8081/
Press 'q' to quit, 's' to save a snapshot.
"""

import cv2
import sys
from datetime import datetime

STREAM_URL = "http://10.250.150.224:8081/"
WINDOW_NAME = "Webstream Viewer"


class OpenCVControl:
    def __init__(self, stream_url: str = STREAM_URL):
        self.stream_url = stream_url
        self._cap: cv2.VideoCapture | None = None
        self._viewing = False
        self._window_name = "Webstream Viewer"

    def connect(self):
        """Open the video stream. Raises RuntimeError if the stream cannot be reached."""
        self._cap = cv2.VideoCapture(self.stream_url)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"Could not open stream at {self.stream_url}. "
                "Check the URL, network connectivity, and that the stream is active."
            )
        print(f"[OK] Connected to {self.stream_url}")

    def disconnect(self):
        """Release the video capture and destroy any open windows."""
        self.stop_view()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            print("[OK] Disconnected.")

    def start_view(self):
        """
        Display the stream in an OpenCV window.
        - Press 'q' to stop viewing.
        - Press 's' to save a snapshot.
        Blocks until the user quits or stop_view() is called internally.
        """
        
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Not connected. Call connect() first.")

        self._viewing = True
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        print(f"[INFO] Viewing stream. Press 'q' to quit, 's' to snapshot.")

        while self._viewing:
            ret, frame = self._cap.read()

            if not ret or frame is None:
                print("[WARN] Failed to grab frame — attempting reconnect...")
                self._cap.release()
                self._cap = cv2.VideoCapture(self.stream_url)
                if not self._cap.isOpened():
                    print("[ERROR] Reconnect failed. Stopping view.")
                    break
                continue

            cv2.imshow(self._window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quit key pressed.")
                break
            elif key == ord("s"):
                self._save_snapshot(frame)

        self.stop_view()

    def stop_view(self):
        """Stop the viewing loop and close the display window."""
        self._viewing = False
        cv2.destroyAllWindows()


    def get_frame(self): # implement
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_snapshot(self, frame):
        filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[SNAP] Saved {filename}")


def main():
    cam = OpenCVControl()
    cam.connect()
    cam.start_view()  # blocks until 'q' is pressed
    cam.disconnect()


if __name__ == "__main__":
    main()