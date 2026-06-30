import socket
import threading
import time
from ImageAnalyzationController import ImageAnalyzationController

class CommandExporter:
    def __init__(self, controller: ImageAnalyzationController, host: str = "0.0.0.0", port: int = 5005):
        self.controller = controller
        self.host = host
        self.port = port
        self.server_socket = None
        self.is_running = False
        self._thread = None

    def start(self):
        """Starts the background listening server thread."""
        self.is_running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(2) # Allow connections
        
        self._thread = threading.Thread(target=self._server_loop, daemon=True)
        self._thread.start()
        print(f"[Exporter Server] Access point online at TCP://{self.host}:{self.port}")

    def _server_loop(self):
        while self.is_running:
            try:
                # Wait for your Control Client script to connect
                client_socket, client_address = self.server_socket.accept()
                print(f"[Exporter Server] Control system connected from: {client_address}")
                
                # Keep streaming current data to this active connection
                while self.is_running:
                    cmd = self.controller.get_latest_command()
                    
                    # Package command with a newline delimiter for easy parsing
                    payload = f"{cmd}\n".encode('utf-8')
                    client_socket.sendall(payload)
                    
                    # Stream command rate matching your control loop demands (e.g. ~33Hz)
                    time.sleep(0.03)
                    
            except (socket.error, ConnectionResetError):
                print("[Exporter Server] Control client disconnected. Waiting for reconnection...")
                try: client_socket.close() 
                except: pass

    def stop(self):
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
        print("[Exporter Server] Offline.")