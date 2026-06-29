import requests
import json
import time

class DriveSystem:
    def __init__(self, base_url):
        self.base_url = base_url

    def control_test(self):
        return True
    
    # --- Stepper aktivieren/deaktivieren ---
    def enable_steppers(self):
        r = requests.post(f"{self.base_url}/api/agv/stepper/enable", json={"stepper": "on"})
        print("Steppers enabled with Code:", r.status_code)
        return r.status_code, r.text
    
    def denable_steppers(self):
        r = requests.post(f"{self.base_url}/api/agv/stepper/enable", json={"stepper": "off"})
        print("Steppers disabled with Code:", r.status_code)
        return r.status_code, r.text

    def drive_control(self, strength_left: float, strength_right: float, max_speed: float = 40.0):
        # speed_left: -1.0 (rückwärts) ... 1.0 (vorwärts)
        # speed_right: -1.0 (rückwärts) ... 1.0 (vorwärts)
        sl = strength_left * max_speed
        sr = strength_right * max_speed
        r = requests.post(f"{self.base_url}/api/agv/stepper/setVelocity", json={"velLeft_perc": sl, "velRight_perc": sr})
        return r.status_code, r.text

    # this function rotates the AGV in place
    def rotate(self, degrees: float = 180.0):
        ROTATION_SPEED = 0.01  # seconds per degree

        r = requests.post(f"{self.base_url}/api/agv/stepper/setVelocity", json={"velLeft_perc": 100, "velRight_perc": -100})
        time.sleep(degrees * ROTATION_SPEED)
        r = requests.post(f"{self.base_url}/api/agv/stepper/setVelocity", json={"velLeft_perc": 0, "velRight_perc": 0})
        
        return r.status_code, r.text