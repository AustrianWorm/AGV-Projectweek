import cv2
from datetime import datetime


class OpenCVControl:
    def __init__(self, stream_url: str):
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

    def disconnect(self):
        """Release the video capture and destroy any open windows."""
        self.stop_view()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            print("[OK] Disconnected.")

    def start_view(self):    
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Not connected. Call connect() first.")

        self._viewing = True
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        while self._viewing:
            ret, frame = self._cap.read()

            if not ret or frame is None:
                print("[WARN] Failed to grab frame — attempting reconnect...")
                self._cap.release()
                self._cap = cv2.VideoCapture(self.stream_url)
                if not self._cap.isOpened():
                    print("[ERROR] Reconnect failed. Stopping view.")
                    self._viewing = False
                    break
                continue

            cv2.imshow(self._window_name, frame)
            key = cv2.waitKey(1) & 0xFF
 
        self.stop_view()

    def stop_view(self):
        self._viewing = False
        cv2.destroyAllWindows()


    def get_frame(self): # implement
        return

def debugCVControl(url):
    cam = OpenCVControl(stream_url=url)
    cam.connect()
    cam.start_view()
    cam.disconnect()