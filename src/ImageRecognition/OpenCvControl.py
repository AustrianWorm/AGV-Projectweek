import cv2
from ImageRecognition.Arucomanager import ArUcoManager, _detect_markers, _make_detector


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

def _annotate_markers(frame, markers):
    """Draw marker outlines and IDs onto frame in-place."""
    for marker in markers.values():
        pts = marker.corners.astype(int)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
        label_pos = (pts[0][0], pts[0][1] - 8)
        cv2.putText(frame, str(marker.id), label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)


def debugCVControl(url: str):
    cam      = OpenCVControl(stream_url=url)
    detector = _make_detector()
    cam.connect()

    cv2.namedWindow("Webstream Viewer", cv2.WINDOW_NORMAL)

    while True:
        frame = cam.get_frame()
        if frame is None:
            continue

        markers = _detect_markers(frame, detector)
        _annotate_markers(frame, markers)

        cv2.imshow("Webstream Viewer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.disconnect()