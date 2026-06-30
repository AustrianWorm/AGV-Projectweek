import cv2
import numpy as np
import threading
import socket
import json
import time

from ImageRecognition.OpenCvControl import OpenCVControl
from ImageRecognition.Arucomanager  import Arucomanager
from ImageRecognition.WallManager   import WallManager

class ImageAnalyzationController:
    def __init__(
        self,
        camera: OpenCVControl,
        aruco:  Arucomanager,
        walls:  WallManager,
        goal_marker_id: int = 10
    ):
        self._camera = camera
        self._aruco  = aruco
        self._walls  = walls
        self._goal_id = goal_marker_id

        self.raw_frame:  np.ndarray | None = None
        self.rect_frame: np.ndarray | None = None
        self.path: list[tuple[float, float]] = []
        
        # Thread-safe variable to hold the calculated command for the exporter
        self.current_command = "STOP"
        self._lock = threading.Lock()

    def start_image_analysis(self) -> bool:
        """Run one full analysis cycle: grab → detect markers → rectify → detect walls → plan path."""
        frame = self._camera.get_frame()
        if frame is None:
            return False
        self.raw_frame = frame

        self._aruco.start_aruco_detection(frame)

        if not self._aruco.is_field_locked():
            with self._lock:
                self.current_command = "STOP"
            return False

        self.rect_frame = self._rectify(frame)
        self._walls.detect_walls(self.rect_frame)
        
        # Calculate the path and update the steering command
        self._compute_path_and_command()
        return True

    def _compute_path_and_command(self):
        """Calculates the A* path and translates it into an actionable command string."""
        agv = self._aruco.get_agv()
        goal = self._aruco.markers.get(self._goal_id)
        
        if agv is None or goal is None:
            self.path = []
            with self._lock:
                self.current_command = "STOP"
            return

        # Find pixel path on the rectified frame
        self.path = self._walls.find_path(agv.center, goal.center, self.rect_frame.shape[:2])
        
        # Determine movement command based on path tracking
        new_command = "STOP"
        
        # If a path exists and we have milestones left to hit
        if self.path and len(self.path) > 1:
            # Look at the next cell waypoint (index 1, since index 0 is our current position)
            target_pt = self.path[1]
            
            # Vector from AGV center to target point
            dx = target_pt[0] - agv.center[0]
            dy = target_pt[1] - agv.center[1]
            
            # Distance check: if we are incredibly close to the goal point, stop or check next
            distance = np.hypot(dx, dy)
            if distance < 15.0: # pixels
                new_command = "STRAIGHT" # Keep coasting or moving to next
            else:
                # Target angle relative to image space grid (OpenCV y goes down, so negate dy)
                target_angle = np.degrees(np.arctan2(-dy, dx))
                
                # Calculate heading error (difference between target angle and AGV face heading)
                # Normalizing the error between -180 and +180 degrees
                angle_error = target_angle - agv.heading
                angle_error = (angle_error + 180) % 360 - 180
                
                # Threshold steering logic
                if angle_error > 20.0:    # Target is significantly to our left
                    new_command = "LEFT"
                elif angle_error < -20.0:  # Target is significantly to our right
                    new_command = "RIGHT"
                else:                      # Target is generally dead ahead
                    new_command = "STRAIGHT"
        else:
            new_command = "STOP"

        # Thread-safely update the exported command
        with self._lock:
            self.current_command = new_command

    def get_latest_command(self) -> str:
        """Exposes the state to the background exporter server thread safely."""
        with self._lock:
            return self.current_command

    def _rectify(self, frame: np.ndarray) -> np.ndarray:
        """Perspective transform wrapper."""
        from ImageRecognition.Arucomanager import _compute_warp
        M = _compute_warp(self._aruco.markers, self._aruco.corner_ids, self._aruco.rect_width, self._aruco.rect_height)
        return cv2.warpPerspective(frame, M, (self._aruco.rect_width, self._aruco.rect_height))