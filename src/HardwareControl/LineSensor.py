import requests
import json

class LineSensor:
    def __init__(self, base_url):
        self.base_url = base_url

    def enable_line_follower(self):
        r = requests.post(f"{self.base_url}/api/agv/linefollower/enable", json={"enable": True})
        print("Line Follower enabled with Code:", r.status_code)
        return r.status_code, r.text
    
    def denable_line_follower(self):
        r = requests.post(f"{self.base_url}/api/agv/linefollower/enable", json={"enable": False})
        print("Line Follower disabled with Code:", r.status_code)
        return r.status_code, r.text


    def set_sample_delta_time(self, delta_time_ms):
        r = requests.post(f"{self.base_url}/api/agv/linefollower/setSampleDeltaTime", json={"sampleDeltaTime": delta_time_ms})
        print("Set Sample Delta Time with Code:", r.status_code)
        return r.status_code, r.text

    def get_sensor_data(self):
        r = requests.get(f"{self.base_url}/api/agv/linefollower/sensors")
        return r.status_code, r.text