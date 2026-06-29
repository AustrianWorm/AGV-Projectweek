from Control.MovementControl import MovementControl
from HardwareControl.Dropper import Dropper

import time
import requests

class AgvControl:
    def __init__(self, ip):
        print("Initializing AGV Control")
        self.base_url = f"http://{ip}"
        self.movement_control = MovementControl(self.base_url)
        self.dropper = Dropper(self.base_url)

        print("Testing connection to AGV...")
        try:
            obj = requests.post(f"{self.base_url}/api/agv/buzzer/beepOnce", json={"toneFrequency_Hz": 200,"duration_ms": 200})
        except Exception as e:
            print("Error connecting to AGV:", e)
        
        print(f"Beep response: {obj.status_code}, {obj.text}")
        
    def run(self):
        print("AGV Control started")
        print(f"Base URL: {self.base_url}")

        self.dropper.up()
        
        i = 0
        NUMBER_OF_STOPS = 7

        while(i < NUMBER_OF_STOPS):
            self.movement_control.follow_line()
            time.sleep(5)
            i += 1

        # alternatives Konzept: alle paths sind nur noch geradeaus,
        # an den Linien wird stehen geblieben und im stand rotiert(Hardcoded)
        # Vorteil von beiden Welten(fahren auf max speed auch möglich) :thinking:

        self.movement_control.rotate(180.0)
        self.dropper.down()
        self.dropper.up()