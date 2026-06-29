from Control.AgvControl import AgvControl

if __name__ == "__main__":
    print("AGV Backend Main gestartet")
    print("Wenn  nichts passiert bist du vll im falschen Wlan ;)")
    
    control = AgvControl(ip="192.168.4.1")
    control.run()