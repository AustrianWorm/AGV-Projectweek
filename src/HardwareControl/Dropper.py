import requests
import json
import time

class Dropper:
    def __init__(self, base_url):
        print("Initializing Dropper Control")
        self.base_url = base_url

    def down(self):
        # implement dropping here
        e = requests.post(f"{self.base_url}/api/agv/pwm/enable", 
        json = {
            "enable": True
        })

        print(e.status_code, e.text)

        sp = requests.post(f"{self.base_url}/api/agv/pwm/channel/setPosition", 
        json = {
            "channel": 1,
            "newPosition": 100
        })

        time.sleep(2)

        print(sp.status_code, sp.text)

        d = requests.post(f"{self.base_url}/api/agv/pwm/enable", 
        json = {
            "enable": False
        })

        print(d.status_code, d.text)

    def up(self):
        # implement up pulling here
        e = requests.post(f"{self.base_url}/api/agv/pwm/enable", 
        json = {
            "enable": True
        })

        print(e.status_code, e.text)

        sp = requests.post(f"{self.base_url}/api/agv/pwm/channel/setPosition", 
        json = {
            "channel": 1,
            "newPosition": 50
        })

        time.sleep(2)

        print(sp.status_code, sp.text)

        d = requests.post(f"{self.base_url}/api/agv/pwm/enable", 
        json = {
            "enable": False
        })

        print(d.status_code, d.text)