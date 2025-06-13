import requests
import time
from datetime import datetime

API_URL = "https://icelec50015.azurewebsites.net/deferables"
OUTPUT_FILE = "deferables_log.txt"
INTERVAL = 300  # seconds between logs

def fetch_and_log():
    try:
        response = requests.get(API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        data = {"error": str(e)}

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | {data}\n"

    with open(OUTPUT_FILE, "a") as f:
        f.write(log_entry)

    print(f"Logged data at {timestamp}")

if __name__ == "__main__":
    print(f"Starting logging from {API_URL} into {OUTPUT_FILE} every {INTERVAL} seconds.")
    try:
        while True:
            fetch_and_log()
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("Logging stopped by user.")
