import sqlite3
import threading
import time
from datetime import datetime

import requests
from flask import Flask, request, jsonify, send_file, Response

import json

import numpy as np
from tensorflow.keras.models import load_model
from resnet_policy import ResNetPolicy
from dqn_resnet_agent import DQNResNetAgent
import joblib

from greedy_schedule import greedy_schedule
from price_threshold_policy import price_threshold_policy
import traceback

import paho.mqtt.client as mqtt

# Initialize Flask app with static folder
app = Flask(
    __name__,
    static_folder='static',    
    static_url_path=''          
)

# When mppt_enabled is False → publish real irradiance from ICElec.
# When mppt_enabled is True → override next publishes: 70 → 100 → random[95–100] …
mppt_enabled = False
mppt_stage = 0  

SCALER_DQN = joblib.load("scaler_dqn.pkl") 
DQN_MODEL = load_model(
    "dqn_resnet_model.keras",
    custom_objects={"ResNetPolicy": ResNetPolicy},
    compile=False
)

# Re‐create the same action space used during training
ACTION_SPACE = np.linspace(-40.0, 40.0, 17)  

PRICE_MODEL = load_model("lstm_model.h5", compile=False)
PRICE_SCALER = joblib.load("scaler.pkl")

DEMAND_MODEL = load_model("demand_model.h5", compile=False)
DEMAND_SCALER = joblib.load("demand_scaler.pkl")

SUN_MODEL = load_model("sun_model.h5", compile=False)
SUN_SCALER = joblib.load("sun_scaler.pkl")


# For ICElec API and fetch interval
BASE_URL = "https://icelec50015.azurewebsites.net"
FETCH_INTERVAL = 5  

hardware_state = {
  'pv_power':   0.0,
  'voltage':    0.0,
  'cap_energy': 0.0,
  'cap_SoC':     0.0,     
  'cap_delta':   0.0,     
  'led1_power': 0.0,
  'led2_power': 0.0,
  'led3_power': 0.0,
  'led4_power': 0.0,
  'import_power':0.0,    
  'export_power':0.0      
}

cost_state = {
    "tick": None,
    "total_cost": 0.0,
    "import_sum": 0.0,
    "export_sum": 0.0
}

last_msg_time = {
    'PV':      None,
    'DC':      None,
    'CAP':     None,
    'EXT':     None,
    'LED1':    None,
    'LED2':    None,
    'LED3':    None,
    'LED4':    None
}
def on_hardware_message(client, userdata, msg):
    topic = msg.topic
    try:
        val = float(msg.payload)
    except:
        return
    now = time.time()

    if topic.endswith("LED1/power"):
        hardware_state['led1_power'] = val
        last_msg_time['LED1'] = now
    elif topic.endswith("LED2/power"):
        hardware_state['led2_power'] = val
        last_msg_time['LED2'] = now
    elif topic.endswith("LED3/power"):
        hardware_state['led3_power'] = val
        last_msg_time['LED3'] = now
    elif topic.endswith("LED4/power"):
        hardware_state['led4_power'] = val
        last_msg_time['LED4'] = now
    elif topic == "PV/power":
        hardware_state['pv_power'] = val
        last_msg_time['PV'] = now
    elif topic == "PV/vbus":
        hardware_state['voltage'] = val
        last_msg_time['DC'] = now               # DC bus is “PV/vbus”
    elif topic == "capacitor/energy":
        hardware_state['cap_energy'] = val
        last_msg_time['CAP'] = now
    elif topic == "capacitor/SoC":
        hardware_state['cap_SoC'] = val
        last_msg_time['CAP'] = now
    elif topic == "capacitor/E_delta":
        hardware_state['cap_delta'] = val
        last_msg_time['CAP'] = now
    elif topic == "External/import_power":
        hardware_state['import_power'] = val
        last_msg_time['EXT'] = now
    elif topic == "External/export_power":
        hardware_state['export_power'] = val
        last_msg_time['EXT'] = now
mqtt_client = mqtt.Client(client_id="server1client", userdata = None, protocol=mqtt.MQTTv5)
mqtt_client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)
mqtt_client.username_pw_set("server1","Password01!")
mqtt_client.on_message = on_hardware_message
mqtt_client.connect("bd7c7a5e1c8e4513ba43dbbb7f288f38.s1.eu.hivemq.cloud", 8883)
mqtt_client.subscribe([
  ("LEDs/LED1/power",0),
  ("LEDs/LED2/power",0),
  ("LEDs/LED3/power",0),
  ("LEDs/LED4/power",0),
  ("PV/power",0),
  ("PV/vbus",0),
  ("capacitor/energy",0),
  ("capacitor/SoC",0),
  ("capacitor/E_delta",0),
  ("External/import_power",0),
  ("External/export_power",0)
])
mqtt_client.loop_start()

# Path to SQLite database file
DB_PATH = "smartgrid.db"

# Database setup (run once at startup)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Table for periodic smart-grid data including day/tick
cursor.execute("""
CREATE TABLE IF NOT EXISTS smart_grid_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day INTEGER,
    tick INTEGER,
    timestamp TEXT,
    sun REAL,
    price_buy REAL,
    price_sell REAL,
    demand REAL,
    pv_power REAL
)
"""
)

# Table for deferrable tasks
cursor.execute("""
CREATE TABLE IF NOT EXISTS deferrables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_tick INTEGER,
    end_tick INTEGER,
    energy REAL,
    fetched_at TEXT
)
"""
)

# Table for yesterday's historical data
cursor.execute("""
CREATE TABLE IF NOT EXISTS yesterday_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick INTEGER,
    buy_price REAL,
    sell_price REAL,
    demand REAL,
    fetched_at TEXT
)
"""
)

# Table for device data
cursor.execute("""
CREATE TABLE IF NOT EXISTS device_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    led1_status TEXT,
    led1_current REAL
)
"""
)

conn.commit()
conn.close()


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_and_store_grid_data():
    global mppt_enabled, mppt_stage
    try:
        price_resp = requests.get(f"{BASE_URL}/price", timeout=5)
        price_resp.raise_for_status()
        pj = price_resp.json()
        day_idx = pj["day"]
        tick_idx = pj["tick"]
        # Endpoint returns 'buy_price' and 'sell_price'
        price_buy = pj.get("buy_price")
        price_sell = pj.get("sell_price")

        sun_resp = requests.get(f"{BASE_URL}/sun", timeout=5)
        sun_resp.raise_for_status()
        sj = sun_resp.json()
        sun_val = sj.get("sun") if isinstance(sj, dict) else sj

        # Decide what to publish under "PV/irradiance" 
        if mppt_enabled:
            if mppt_stage == 0:
                publish_val = 70
                mppt_stage = 1
            elif mppt_stage == 1:
                publish_val = 100
                mppt_stage = 2
            else:
                publish_val = random.randint(95, 100)
            mqtt_client.publish("PV/irradiance", str(publish_val))
            print(f"[MPPT OVERRIDE] Published irradiance: {publish_val} to PV/irradiance")
        else:
            mqtt_client.publish("PV/irradiance", str(sun_val))
            print(f"Published irradiance: {sun_val} to PV/irradiance")

        demand_resp = requests.get(f"{BASE_URL}/demand", timeout=5)
        demand_resp.raise_for_status()
        dj = demand_resp.json()
        demand_val = dj.get("demand") if isinstance(dj, dict) else dj

        db = get_db_connection()
        cur = db.cursor()
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        # Insert today's live data
        cur.execute(
            "INSERT INTO smart_grid_data (day, tick, timestamp, sun, price_buy, price_sell, demand, pv_power) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (day_idx,
            tick_idx,
            ts,
            float(sun_val),
            float(price_buy),
            float(price_sell),
            float(demand_val),
            float(hardware_state.get('pv_power', 0.0))
            )
        )
        db.commit()
        global cost_state
        if cost_state["tick"] != tick_idx:
            # Tick changed
            if cost_state["tick"] is not None:
                tick_revenue = cost_state["export_sum"] * price_buy
                tick_expense = cost_state["import_sum"] * price_sell
                tick_net = tick_revenue - tick_expense
                cost_state["total_cost"] += tick_net

            # Reset accumulators
            if tick_idx == 0:
                cost_state["total_cost"] = 0.0  
                # showing a new day
            cost_state["tick"] = tick_idx
            cost_state["import_sum"] = 0.0
            cost_state["export_sum"] = 0.0

        # Accumulate during same tick
        cost_state["import_sum"] += hardware_state["import_power"]
        cost_state["export_sum"] += hardware_state["export_power"]

        # On rollover to tick 0, archive full yesterday into yesterday_data
        if tick_idx == 0:
            prev_day = day_idx - 1
            cur.execute("DELETE FROM yesterday_data")
            rows = db.execute(
                "SELECT tick, price_buy, price_sell, demand FROM smart_grid_data WHERE day = ? ORDER BY tick",
                (prev_day,)
            ).fetchall()
            for r in rows:
                cur.execute(
                    "INSERT INTO yesterday_data (tick, buy_price, sell_price, demand, fetched_at) VALUES (?, ?, ?, ?, ?)",
                    (r['tick'], r['price_buy'], r['price_sell'], r['demand'], ts)
                )
            db.commit()
        db.close()
        print(f"[day={day_idx} tick={tick_idx}] sun={sun_val}, buy={price_buy}, sell={price_sell}, demand={demand_val}")
    except Exception as e:
        print(f"Error fetching/storing grid data: {e}")

# Periodic fetch: ICElec deferrables 
def fetch_and_store_deferrables():
    try:
        ext = requests.get(f"{BASE_URL}/deferables", timeout=5)
        ext.raise_for_status()
        arr = ext.json()   
        db = get_db_connection()
        db.execute("DELETE FROM deferrables")
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        for item in arr:
            db.execute(
              "INSERT INTO deferrables(start_tick,end_tick,energy,fetched_at) VALUES(?,?,?,?)",
              ( int(item["start"]),
                int(item["end"]),
                float(item["energy"]),
                ts )
            )
        db.commit()
        db.close()
    except Exception as e:
        print("Deferrables fetch/store error:", e)

def agent_act(state_window):
    """
    state_window: np.array of shape (60, 4), dtype=float32, already scaled.
    We need to add batch‐dim → (1, 60, 4) and do model.predict to get q‐values.
    Then pick argmax over the 17 Q‐values → index.
    """
    x = np.expand_dims(state_window.astype(np.float32), axis=0)  # shape (1,60,4)
    q_vals = DQN_MODEL.predict(x, verbose=0)[0]  # shape (17,)
    action_index = int(np.argmax(q_vals))
    return action_index


def simulate_decision(pred_sun, pred_buy, pred_sell, pred_demand):
    """
    Given four arrays of shape (60,), form a 60×4 matrix:
        features[i] = [pred_sun[i], pred_price_buy[i], pred_price_sell[i], pred_demand[i]]
    Then scale with SCALER_DQN.transform(...).
    Finally, for each tick j from 0..59, 
    build a 60×4 “state window” by zero‐padding at the top so that
    state_window[j] always has length 60. Call agent_act(...) 
    to get an action index, convert to energy (J), and record it.
    """
    LOOK_BACK = 60
    # Build the raw features matrix (60 × 4):
    raw = np.stack([pred_sun, pred_buy, pred_sell, pred_demand], axis=1)  # (60,4)
    scaled = SCALER_DQN.transform(raw)                                    # still (60,4)

    ticks = []
    decision_energy = []
    for j in range(LOOK_BACK):
        #    form a 60×4 “window” with zero‐padding at the front:
        #    if j=0: window = scaled[0:60]
        #    if j>0: shift up by j, pad last j rows with zeros.
        if j == 0:
            state_window = scaled.copy()                       # shape (60,4)
        else:
            # shift feature rows “up” so that the oldest is at top:
            window_data = scaled[j:LOOK_BACK]                  # shape (60−j,4)
            padding = np.zeros((j, 4), dtype=scaled.dtype)     # shape (j,4)
            state_window = np.vstack([window_data, padding])   # shape (60,4)

        # 2) ask the DQN‐policy for an action index:
        a_idx = agent_act(state_window)                        # integer 0..16
        power_w = float(ACTION_SPACE[a_idx])                


        ticks.append(int(j))
        decision_energy.append(float(power_w))

    return ticks, decision_energy

def fetch_grid_data_periodically():
    while True:
        fetch_and_store_grid_data()
        time.sleep(FETCH_INTERVAL)

thread = threading.Thread(target=fetch_grid_data_periodically, daemon=True)
thread.start()

# --- API Endpoints ---

@app.route('/status/decision', methods=['GET'])
def get_decision():
    """
    1) Call the same logic that /status/predict uses to get predictions.
    2) Pass them to simulate_decision(...) to produce ticks+decision_energy.
    3) Return JSON: {"ticks": [...], "decision_energy": [...] }
    """
    try:
        # replicate /status/predict internals (pull yesterday, scale, predict):
        LOOK_BACK = 60

        # fetch yesterday’s 60 rows of buy/sell/demand via ICElec /yesterday
        resp = requests.get(f"{BASE_URL}/yesterday", timeout=10)
        resp.raise_for_status()
        yesterday = resp.json()
        price_arr = np.array([[r['buy_price'], r['sell_price']] for r in yesterday])
        demand_arr = np.array([[r['demand']] for r in yesterday])

        # pull “sun” from our smart_grid_data for previous day
        db = get_db_connection()
        latest = db.execute("SELECT day FROM smart_grid_data ORDER BY id DESC LIMIT 1").fetchone()
        prev_day = latest['day'] - 1
        sun_rows = db.execute(
            "SELECT sun FROM smart_grid_data WHERE day = ? ORDER BY tick",
            (prev_day,)
        ).fetchall()
        db.close()
        suns = [r['sun'] for r in sun_rows]
        if len(suns) < LOOK_BACK:
            suns = [0.0] * (LOOK_BACK - len(suns)) + suns
        sun_arr = np.array(suns).reshape(LOOK_BACK, 1)

        # price forecast (bidirectional LSTM)
        sp = PRICE_SCALER.transform(price_arr)                        # from /status/predict
        pp = PRICE_MODEL.predict(sp.reshape(1, LOOK_BACK, 2))[0]
        buy_vals, sell_vals = zip(*PRICE_SCALER.inverse_transform(pp))

        # demand forecast
        sd = DEMAND_SCALER.transform(demand_arr)
        pd_d = DEMAND_MODEL.predict(sd.reshape(1, LOOK_BACK, 1))[0]
        preds_demand = DEMAND_SCALER.inverse_transform(pd_d.reshape(-1, 1)).flatten()

        # sun forecast
        ss = SUN_SCALER.transform(sun_arr)
        ps = SUN_MODEL.predict(ss.reshape(1, LOOK_BACK, 1))[0]
        preds_sun = SUN_SCALER.inverse_transform(ps.reshape(-1, 1)).flatten()

        # Convert everything to numpy arrays of shape (60)
        pred_buy = np.array([float(v) for v in buy_vals])
        pred_sell = np.array([float(v) for v in sell_vals])
        pred_demand = np.array([float(v) for v in preds_demand])
        pred_sun = np.array([float(v) for v in preds_sun])

        # Run DQN agent simulation over these 60‐tick predictions:
        ticks, decision_energy = simulate_decision(pred_sun, pred_buy, pred_sell, pred_demand)

        # Return native Python lists of floats/ints
        return jsonify({
            "ticks": [int(t) for t in ticks],
            "decision_energy": [float(e) for e in decision_energy]
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("Error in /status/decision:\n", tb)
        return jsonify({
            "error": str(e),
            "trace": tb
        }), 500

@app.route('/status/control', methods=['GET'])
def get_control_status():
    # Fetch latest irradiance
    db = get_db_connection()
    row = db.execute(
        "SELECT sun FROM smart_grid_data ORDER BY id DESC LIMIT 1"
    ).fetchone()
    irradiance = row["sun"] if row else 0.0

    latest = db.execute("""
        SELECT tick,
               price_buy,
               price_sell,
               demand
          FROM smart_grid_data
         ORDER BY id DESC
         LIMIT 1
    """).fetchone()
    db.close()

    current_tick       = latest["tick"]       if latest else 0
    current_price_buy  = latest["price_buy"]  if latest else 0.0
    current_price_sell = latest["price_sell"] if latest else 0.0
    inst_demand_w      = float(latest["demand"]) if latest else 0.0

    now = time.time()
    THRESHOLD = 10
    def is_connected(key):
        ts = last_msg_time.get(key)
        return (ts is not None) and (now - ts < THRESHOLD)

    connected = {
        'PV':   is_connected('PV'),
        'DC':   is_connected('DC'),
        'CAP':  is_connected('CAP'),
        'EXT':  is_connected('EXT'),
        'LED1': is_connected('LED1'),
        'LED2': is_connected('LED2'),
        'LED3': is_connected('LED3'),
        'LED4': is_connected('LED4')
    }

    # Fetch the deferrable schedule and sum joules at this tick
    sched = requests.get("http://localhost:5000/status/schedule").json()
    plans = sched.get('plans', [])
    total_def_j = sum(
        alloc
        for plan in plans
        for (t, alloc) in plan
        if t == current_tick
    )
    # Convert joules→watts over 5s tick and add instantaneous demand
    def_w   = total_def_j / 5.0
    combined_w = inst_demand_w + def_w
    # Split evenly across 4 LEDs with 1W cap
    per_led = min(combined_w / 4.0, 1.0)

    auto_powers = {
        'auto_led1_power': per_led,
        'auto_led2_power': per_led,
        'auto_led3_power': per_led,
        'auto_led4_power': per_led,
    }
    # end auto setpoints calc 

    # 4) Package JSON response 
    data = hardware_state.copy()
    data.update(auto_powers)
    data.update({
        'irradiance':   irradiance,
        'connected':    connected,
        'tick':         current_tick,
        'price_buy':    current_price_buy,
        'price_sell':   current_price_sell,
        'mppt_enabled': mppt_enabled,
        'cost':         round(cost_state["total_cost"] / 100.0, 2)
    })
    action_live = price_threshold_policy(
        [[irradiance, float(latest["demand"]), current_price_buy, current_price_sell]],
        t=-1
    )
    data['action_live'] = action_live

    return jsonify(data), 200



@app.route('/controls/override', methods=['POST'])
def manual_override():
    data = request.get_json()
    for led in ['led1_power','led2_power','led3_power','led4_power']:
        if led in data:
            channel = led.split('_')[0].upper()
            mqtt_client.publish(f"LEDs/{channel}/setpoint", str(float(data[led])))
            print(f"sending data..")
    mqtt_client.publish("sync", "1")
    return ('', 204)

@app.route('/controls/capacitor', methods=['POST'])
def control_capacitor():
    data = request.get_json()
    cmd = data.get('command', '').strip().upper()
    if cmd in ("S", "E", "H"):
        mqtt_client.publish("capacitor/cmd", cmd)
        print(f"Publishing capacitor command to MQTT: {cmd}")
        return jsonify({"status": "sent", "command": cmd}), 200
    else:
        return jsonify({"status": "invalid", "command": cmd}), 400

@app.route('/controls/mppt', methods=['POST'])
def control_mppt1():
    global mppt_enabled, mppt_stage

    data = request.get_json()
    cmd  = data.get('command')

    if cmd == 1:
        mppt_enabled = True
        mppt_stage   = 0
    elif cmd == 0:
        mppt_enabled = False
    else:
        return jsonify({ "status": "invalid", "command": cmd }), 400
    mqtt_client.publish("PV/mode", str(cmd))
    print(f"Published MPPT command: {cmd}; mppt_enabled={mppt_enabled}")
    return jsonify({ "status": "sent", "command": cmd }), 200


@app.route('/status/grid', methods=['GET'])
def get_grid_status():
    # Fetch latest grid snapshot
    db = get_db_connection()
    latest = db.execute(
        "SELECT * FROM smart_grid_data ORDER BY id DESC LIMIT 1"
    ).fetchone()
    current_day  = latest['day']
    current_tick = latest['tick']

    # Build history for the day 
    rows = db.execute("""
        SELECT tick, sun, price_buy, price_sell, demand, pv_power
          FROM smart_grid_data
         WHERE day = ?
         ORDER BY tick
    """, (current_day,)).fetchall()
    db.close()

    # Fetch deferrable schedule & compute extra load 
    sched = requests.get("http://localhost:5000/status/schedule").json()
    plans = sched.get('plans', [])
    # sum all joules scheduled for this tick
    total_def_j = sum(
        alloc
        for plan in plans
        for (t, alloc) in plan
        if t == current_tick
    )
    # convert J→W over the 5 s tick
    def_w = total_def_j / 5.0
    def_map = {}
    for plan in plans:
      for (t, alloc_j) in plan:
        def_map[t] = def_map.get(t, 0) + alloc_j

    # Compute combined demand and per-LED split 
    inst_w   = latest['demand']
    combined = inst_w + def_w
    per_led  = min(combined / 4.0, 1.0)  # cap at 1 W each
    def_w    = def_map.get(current_tick, 0) / 5.0

   

    # Assemble and return JSON response  
    history = []
    for r in rows:
      t     = r['tick']
      inst  = r['demand']
      dw    = def_map.get(t, 0) / 5.0
      comb  = inst + dw
      history.append({
        'tick':      t,
        'sun':       r['sun'],
        'price_buy': r['price_buy'],
        'price_sell':r['price_sell'],
        'demand':    inst,
        'def_w':     round(dw,2),           
        'combined':  round(comb,2),         
        'pv_power':  r['pv_power']
      })
    response = {
        "day":          current_day,
        "tick":         current_tick,
        "price_buy":    latest["price_buy"],
        "price_sell":   latest["price_sell"],
        "sun":          latest["sun"],
        "demand":       inst_w,               
        "deferrable":   total_def_j,          
        "def_w":        round(def_w, 2),      
        "combined":     round(combined, 2),   
        "led1_power":   per_led,
        "led2_power":   per_led,
        "led3_power":   per_led,
        "led4_power":   per_led,
        "pv_power":     hardware_state["pv_power"],
        "connected_pv": (
            last_msg_time["PV"] is not None
            and (time.time() - last_msg_time["PV"] < 10)
        ),
        "cost":         round(cost_state["total_cost"] / 100.0, 2),
        'def_w':     round(def_w,2),
        'combined':  round(combined,2),
        'history':   history
    }

    return jsonify(response), 200

@app.route('/status/days', methods=['GET'])
def list_days():
    db = get_db_connection()
    rows = db.execute(
       "SELECT DISTINCT day FROM smart_grid_data ORDER BY day DESC"
    ).fetchall()
    db.close()
    return jsonify([r['day'] for r in rows]), 200

@app.route('/status/history', methods=['GET'])
def history_for_day():
    day = request.args.get('day')
    if not day:
        return jsonify({"error":"Missing 'day' parameter"}), 400
    db = get_db_connection()
    rows = db.execute(
        "SELECT tick, price_buy AS buy_price, price_sell AS sell_price, demand, sun AS irradiance "
        "  FROM smart_grid_data WHERE day = ? ORDER BY tick",
        (int(day),)
    ).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows]), 200

@app.route('/status/history/days')
def history_days():
    limit = request.args.get('limit', default=100, type=int)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
      SELECT DISTINCT day
        FROM smart_grid_data
       ORDER BY day DESC
       LIMIT ?
    """, (limit,))
    days = [row['day'] for row in cur.fetchall()]
    conn.close()
    return jsonify(days)

@app.route('/status/irradiance', methods=['GET'])
def get_irradiance():
    db = get_db_connection()
    row = db.execute(
        "SELECT sun FROM smart_grid_data ORDER BY id DESC LIMIT 1"
    ).fetchone()
    db.close()

    if not row:
        return jsonify({"irradiance": 0.0}), 200

    return jsonify({"irradiance": row["sun"]}), 200

@app.route('/status/deferrables', methods=['GET'])
def get_deferrables():
    db = get_db_connection()
    rows = db.execute(
        "SELECT start_tick, end_tick, energy FROM deferrables ORDER BY id"
    ).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows]), 200

@app.route('/status/predict', methods=['GET'])
def get_price_forecast():
    try:
        LOOK_BACK = 60

        # 1) Buy/Sell/Demand from ICElec’s /yesterday
        resp = requests.get(f"{BASE_URL}/yesterday", timeout=10)
        resp.raise_for_status()
        yesterday = resp.json()
        price_arr  = np.array([[r['buy_price'], r['sell_price']] for r in yesterday])
        demand_arr = np.array([[r['demand']] for r in yesterday])

        # 2) Sun from your own smart_grid_data table
        db = get_db_connection()
        latest = db.execute("SELECT day FROM smart_grid_data ORDER BY id DESC LIMIT 1").fetchone()
        prev_day = latest['day'] - 1
        sun_rows = db.execute(
            "SELECT sun FROM smart_grid_data WHERE day = ? ORDER BY tick",
            (prev_day,)
        ).fetchall()
        db.close()

        suns = [r['sun'] for r in sun_rows]
        if len(suns) < LOOK_BACK:
            suns = [0.0] * (LOOK_BACK - len(suns)) + suns
        sun_arr = np.array(suns).reshape(LOOK_BACK, 1)

        # 3) Price forecast
        sp = PRICE_SCALER.transform(price_arr)
        pp = PRICE_MODEL.predict(sp.reshape(1, LOOK_BACK, 2))[0]
        buy_vals, sell_vals = zip(*PRICE_SCALER.inverse_transform(pp))

        # 4) Demand forecast
        sd = DEMAND_SCALER.transform(demand_arr)
        pd_d = DEMAND_MODEL.predict(sd.reshape(1, LOOK_BACK, 1))[0]
        preds_demand = DEMAND_SCALER.inverse_transform(pd_d.reshape(-1,1)).flatten().tolist()

        # 5) Sun forecast
        ss = SUN_SCALER.transform(sun_arr)
        ps = SUN_MODEL.predict(ss.reshape(1, LOOK_BACK, 1))[0]
        preds_sun = SUN_SCALER.inverse_transform(ps.reshape(-1,1)).flatten().tolist()

        # Cast everything to native Python types:
        py_buy  = [float(v) for v in buy_vals]
        py_sell = [float(v) for v in sell_vals]
        py_dem  = [float(v) for v in preds_demand]
        py_sun  = [float(v) for v in preds_sun]

        # Return only Python primitives
        return jsonify({
            "start_tick":    0,         # native int
            "future_buy":    py_buy,    # list of Python floats
            "future_sell":   py_sell,
            "future_demand": py_dem,
            "future_sun":    py_sun,
            "timestamp":     datetime.utcnow().isoformat()
        })
       

    except Exception as e:
        # print full traceback to console
        import traceback
        tb = traceback.format_exc()
        print(tb)
        # and return it in JSON so the browser can show it
        return jsonify({
          "error": str(e),
          "trace": tb
        }), 500




@app.route('/status/predict_debug', methods=['GET'])
def predict_debug():
    LOOK_BACK = 60

    # Pull yesterday’s 60 rows in one go
    resp = requests.get(f"{BASE_URL}/yesterday", timeout=10)
    resp.raise_for_status()
    yesterday = resp.json()

    arr = np.array([[r['buy_price'], r['sell_price']] for r in yesterday])
    scaled      = PRICE_SCALER.transform(arr)
    pred_scaled = PRICE_MODEL.predict(scaled.reshape(1, LOOK_BACK, 2))[0]
    pred        = PRICE_SCALER.inverse_transform(pred_scaled)

    return jsonify({
        "history":    arr.tolist(),
        "scaled":     scaled.tolist(),
        "prediction": pred.tolist()
    })


@app.route('/predict_view')
def predict_view():
    try:
        # Call the debug endpoint function directly
        resp = predict_debug()
        raw = resp.get_data(as_text=True)
        data = json.loads(raw)
        pretty = json.dumps(data, indent=2)
        return f"<html><body><pre>{pretty}</pre></body></html>"
    except Exception as e:
        # Log the traceback to console and show it in browser
        tb = traceback.format_exc()
        print("Error in /predict_view:\n", tb)
        return f"<html><body><h1>Exception in predict_view</h1><pre>{tb}</pre></body></html>", 500


@app.route('/status/schedule', methods=['GET'])
def get_deferrables_schedule():
    """
    Returns the 60‐tick schedule for each of the 3 deferrable tasks.
    """
    # 1️ Fetch raw deferrable tasks from ICElec
    resp = requests.get(f"{BASE_URL}/deferables", timeout=5)
    resp.raise_for_status()
    raw_tasks = resp.json()  # [{ "start": 10, "end": 20, "energy": 1234.5 }, ...]
    tasks = [(t["energy"], int(t["start"]), int(t["end"])) for t in raw_tasks]

    # 2️ Pull 60‐tick predictions via /status/predict logic
    pred = requests.get("http://localhost:5000/status/predict").json()
    LOOK_BACK = 60
    PV      = np.array(pred["future_sun"])
    D_fixed = np.array(pred["future_demand"])
    P_buy   = np.array(pred["future_buy"])
    P_sell  = np.array(pred["future_sell"])

    # 3️ Run the greedy scheduler
    D_def_profile, plans = greedy_schedule(PV, D_fixed, P_buy, P_sell, tasks)

    # 4 Return JSON
    return jsonify({
      "tick_profile": D_def_profile.tolist(),  # length‐60 array of power
      "plans":        plans                   
    }), 200

@app.route('/status/policy', methods=['POST'])
def get_policy():
    """
    Expects JSON {"state_seq": [[sun,demand,buy_price,sell_price],...], "future_seq": [[...],...]}
    Returns {"actions": [...], "future_actions": [...]}
    """
    body      = request.get_json()
    state_seq = body['state_seq']
    future_seq= body.get('future_seq', [])
    action_live = price_threshold_policy(state_seq, t=-1)

    future_actions = []
    for i in range(len(future_seq)):
        # i=0 → no preds → same as action_live
        # i>0 → add the first i predictions on top of state_seq
        seq = state_seq + future_seq[:i]
        future_actions.append(price_threshold_policy(seq, t=-1))

    return jsonify({
        'action_live': action_live,
        'future_actions': future_actions
    }), 200


@app.route('/download/history', methods=['GET'])
def download_history_txt():
    day = request.args.get('day')
    if not day:
        return "Missing 'day' parameter", 400
    db = get_db_connection()
    rows = db.execute(
        "SELECT tick, price_buy AS buy_price, price_sell AS sell_price, demand "
        "  FROM smart_grid_data "
        " WHERE day = ? "
        " ORDER BY tick", (int(day),)
    ).fetchall()
    db.close()

    lines = ['tick buy_price sell_price demand']
    for r in rows:
        lines.append(f"{r['tick']} {r['buy_price']} {r['sell_price']} {r['demand']}")
    txt = "\n".join(lines)

    return Response(
        txt,
        mimetype='text/plain; charset=utf-8',
        headers={
            'Content-Disposition': f'attachment; filename=history_{day}.txt'
        }
    )

@app.route('/download/all', methods=['GET'])
def download_all_history_txt():
    db = get_db_connection()
    # grab every tick from every day, in day→tick order
    rows = db.execute("""
      SELECT tick,
             price_buy   AS buy_price,
             price_sell  AS sell_price,
             demand
        FROM smart_grid_data
       ORDER BY day, tick
    """).fetchall()
    db.close()

    lines = ['tick buy_price sell_price demand']
    for r in rows:
        lines.append(f"{r['tick']} {r['buy_price']} {r['sell_price']} {r['demand']}")
    txt = "\n".join(lines)

    return Response(
        txt,
        mimetype='text/plain; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename=all_history.txt'}
    )

@app.route('/status/device', methods=['GET'])
def get_device_status():
    db = get_db_connection()
    row = db.execute("SELECT * FROM device_data ORDER BY id DESC LIMIT 1").fetchone()
    db.close()
    if not row:
        return jsonify({"error": "No device data available"}), 404
    return jsonify({
        "timestamp": row["timestamp"],
        "led1_status": row["led1_status"],
        "led1_current": row["led1_current"]
    }), 200

@app.route('/update/device', methods=['GET'])
def update_device():
    status = request.args.get('led1_status')
    current = request.args.get('led1_current')
    if status is None or current is None:
        return jsonify({"error": "Missing led1_status or led1_current"}), 400
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    db = get_db_connection()
    db.execute(
        "INSERT INTO device_data (timestamp, led1_status, led1_current) VALUES (?, ?, ?)",
        (timestamp, status, float(current))
    )
    db.commit()
    db.close()
    return jsonify({"message": "Device data stored successfully"}), 200

def schedule_deferrables_fetch():
    fetch_and_store_deferrables()
    threading.Timer(FETCH_INTERVAL, schedule_deferrables_fetch).start()

schedule_deferrables_fetch()

@app.route('/')
def dashboard():
    return app.send_static_file('index.html')




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
