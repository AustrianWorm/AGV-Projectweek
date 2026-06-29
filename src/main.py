from Control.AgvControl import AgvControl
from ImageRecognition.OpenCvControl import debugCVControl

import argparse
import json
with open("config.json") as f:
    config = json.load(f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["debug-control", "debug-image", "release"], default="release")
    args = parser.parse_args()

    if(args.mode == "debug-control"):
        print("Started in Debug Control Mode")
        control = AgvControl(url=config["agv"]["agv_url"])
        control.run()
    if(args.mode == "debug-image"):
        print("Started in Debug Image Mode")
        debugCVControl(url=config["camera"]["stream_url"])
    elif(args.mode == "release"):
        print("Started in Release Mode")


