from Control.MovementControl import MovementControl

import time
import requests

class AgvControl:
    def __init__(self, ip):
        print("Initializing AGV Control")
        self.base_url = f"http://{ip}"
        self.movement_control = MovementControl(self.base_url)

        print("Testing connection to AGV...")
        try:
            obj = requests.post(f"{self.base_url}/api/agv/buzzer/beepOnce", json={"toneFrequency_Hz": 200,"duration_ms": 200})
        except Exception as e:
            print("Error connecting to AGV:", e)
        
        print(f"Beep response: {obj.status_code}, {obj.text}")
        
    def run(self):
        print("AGV Control started")
        print(f"Base URL: {self.base_url}")

        self.movement_control.rotate(180.0)