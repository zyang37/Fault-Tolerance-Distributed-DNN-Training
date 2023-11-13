import multiprocessing
import time

class Monitor:
    def __init__(self):
        self.process = None
        self.exit_event = multiprocessing.Event()
        self.data_queue = multiprocessing.Queue()

    def start(self):
        self.process = multiprocessing.Process(target=self.run)
        self.process.start()

    def run(self):
        while not self.exit_event.is_set():
            if not self.data_queue.empty():
                data = self.data_queue.get()
                print(f"Received data: {data}")
                # Process the data here
            else:
                time.sleep(1)  # Sleep if there's no data

    def stop(self):
        self.exit_event.set()
        self.process.join()

    def send_data(self, data):
        self.data_queue.put(data)

    def is_running(self):
        return self.process is not None and self.process.is_alive()
    
    
if __name__ == '__main__':    
    # Usage
    monitor = Monitor()
    monitor.start()

    try:
        # Main process does its work and sends data to the monitor
        for i in range(5):
            monitor.send_data(f"Data {i}")
            time.sleep(1)
    finally:
        monitor.stop()
        print("Monitor stopped.")
        