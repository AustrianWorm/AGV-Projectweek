from Control.ControlBackends.MovementControl import MovementControl
from HardwareBackends.DriveSystem import DriveSystem
from HardwareBackends.LineSensor import LineSensor

import requests

class AgvControl:
    def __init__(self, url):
        print("Initializing AGV Control")
        self.url = url

        print("Testing connection to AGV...")
        try:
            requests.post(f"{self.base_url}/api/agv/buzzer/beepOnce", json={"toneFrequency_Hz": 200,"duration_ms": 200})
            print("Connection success")
        except Exception as e:
            print("Error connecting to AGV, read Documentation:", e)

        self.movement_control = MovementControl(self.url)
        
    def run(self):
        print("AGV run")
        self.movement_control.rotate(180.0)