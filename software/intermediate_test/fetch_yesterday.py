#!/usr/bin/env python3
"""
fetch_yesterday_txt.py

Fetch the full “yesterday” dataset from ICElec and save it as a plain-text .txt file.
Each line has: tick buy_price sell_price demand
"""

import requests
from datetime import datetime, timedelta

# Base ICElec API URL
BASE_URL = "https://icelec50015.azurewebsites.net"

def fetch_yesterday_to_txt():
    # 1) Fetch yesterday's data
    url = f"{BASE_URL}/yesterday"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()  # Expect list of {tick, buy_price, sell_price, demand}

    # 2) Determine filename based on yesterday's date
    yesterday = datetime.utcnow().date() - timedelta(days=1)
    date_str  = yesterday.strftime("%Y-%m-%d")
    txt_file  = f"yesterday_{date_str}.txt"

    # 3) Write plain-text table
    # First line: header
    # Following lines: space-separated values
    with open(txt_file, "w") as f:
        f.write("tick buy_price sell_price demand\n")
        for rec in data:
            line = f"{rec.get('tick', '')} {rec.get('buy_price', '')} " \
                   f"{rec.get('sell_price', '')} {rec.get('demand', '')}\n"
            f.write(line)

    print(f"✔ Saved TXT to {txt_file}")

if __name__ == "__main__":
    try:
        fetch_yesterday_to_txt()
    except Exception as e:
        print("Error fetching yesterday data:", e)
