from HardwareControl.LineSensor import LineSensor
from HardwareControl.DriveSystem import DriveSystem

import json
import time
import math

COLOR_VALUE_MIN = 100

SENSOR_SENSE_PAUSE = 0.03
SPEED_FACTOR = 2.5

# diese Funktion muss überarbeitet werden
def calculate_direction(sensor_data):
    weights = [-2, -1, 0, 1, 2]

    adjusted = [v if v >= COLOR_VALUE_MIN else 0 for v in sensor_data]

    total = sum(adjusted)

    raw = sum(w * v for w, v in zip(weights, adjusted)) / total
    direction_x = raw / 2.0

    print(direction_x)
    return  max(-1.0, min(1.0, direction_x))

# besser algorithmus um stop linie zu erkennen notwendig
def is_stop_line(sensor_data):
    return sum(sensor_data) <= 200

class MovementControl:
    def __init__(self, base_url):
        print("Initializing Movement Control")
        self.base_url = base_url
        
    def follow_line(self):
        print("Starting line following mode")
        line_sensor = LineSensor(base_url=self.base_url)
        drive_system = DriveSystem(base_url=self.base_url)
        
        line_sensor.enable_line_follower()
        drive_system.enable_steppers()
        
        drive_system.drive_control(0,0)

        while True:
            status_code, text = line_sensor.get_sensor_data()
            jsonobj = json.loads(text) 
            sensor_data = jsonobj["data"]["sensors"] # array of 5 light sensor values
        
            v = calculate_direction(sensor_data)

            if is_stop_line(sensor_data):
                print("Stop detected")
                drive_system.drive_control(0, 0)
                break

            direction_x = 1.0
            direction_y = 1.0

            if v > 0:
                direction_x = v * SPEED_FACTOR
                print("rechts")
            elif v < 0:
                direction_y = abs(v) * SPEED_FACTOR
                print("links")
            else:
                print("gerade aus")

            drive_system.drive_control(strength_left=direction_x, strength_right=direction_y,max_speed=40.0)

            time.sleep(SENSOR_SENSE_PAUSE)
        return
    
    def straight_until_line(self):
        print("Driving straight until line is detected")
        line_sensor = LineSensor(base_url=self.base_url)
        drive_system = DriveSystem(base_url=self.base_url)
        
        line_sensor.enable_line_follower()
        drive_system.enable_steppers()
        
        drive_system.drive_control(0,0)

        while True:
            status_code, text = line_sensor.get_sensor_data()
            jsonobj = json.loads(text) 
            sensor_data = jsonobj["data"]["sensors"] # array of 5 light sensor values

            if is_stop_line(sensor_data):
                print("Line detected")
                drive_system.drive_control(0, 0)
                break

            drive_system.drive_control(strength_left=1.0, strength_right=1.0, max_speed=40.0)

            time.sleep(SENSOR_SENSE_PAUSE)
        return

    def rotate(self, degrees: float = 180.0):
        print(f"Rotating AGV by {degrees} degrees")
        return DriveSystem(base_url=self.base_url).rotate(degrees=degrees)