# Manual MQTT client for testing latency
#
# This program will send a custom MQTT command every 5 seconds by default.
# It will also print out the exact time (in nanoseconds) the command was sent.
#
# Instructions:
# 1. Install "paho-mqtt" module on your laptop using pip on the command line ("pip install paho-mqtt")
# 2. Open this program on your laptop (not in Thonny because Thonny can only run one program at once)
# 3. Change INTERVAL, TOPIC and MESSAGE variables to your preferred value for testing
# 4. Change SMPS Pico code so that it prints out time.time_ns() inside the on_message function
# 5. Run this program on your laptop (not in Thonny)
# 6. Then run the SMPS Pico program in Thonny so that you can see what it is printing out
# 7. Once enough times have been printed, find the difference between each number printed by this program,
#    and each number printed by the SMPS program (they should be in the same order)
#
# NOTE: the SMPS may have its timing off by 3600 seconds (an hour) because of british summer time

INTERVAL = 5 # Time interval in seconds between each message being sent
TOPIC = "capacitor/cmd" # Topic to send message on
MESSAGE = "H" # Message to send to SMPS

import time
import paho.mqtt.client as mqtt

def on_message(client, userdata, message):
    m = message.payload.decode("utf-8")
    print("Message on "+message.topic+": "+m)
    global setpoint
    global next_setpoint
    
    if message.topic == "sync":
        setpoint = next_setpoint
    else:
        next_setpoint = m

setpoint = 0
next_setpoint = 0

broker_addr = "bd7c7a5e1c8e4513ba43dbbb7f288f38.s1.eu.hivemq.cloud"
broker_port = 8883

client = mqtt.Client(
    client_id = "LATENCY_TEST",
    userdata = None,
    protocol = mqtt.MQTTv5
)
client.on_message = on_message

client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)
client.username_pw_set("server1", "Password01!")

start_time = time.time()
print("Connecting to MQTT server... ",end="")
try:
    client.connect(broker_addr, port=broker_port)
except ConnectionRefusedError:
    print("\nERROR: Client could not connect")
    exit(0)
time_taken = time.time() - start_time
print(f"Done! (took {time_taken:.3f} seconds)")

client.loop_start()
client.subscribe("LEDs/LED1/setpoint")
client.subscribe("sync")

while True:
    time.sleep(INTERVAL)
    client.publish(TOPIC, MESSAGE)
    print(time.time_ns())

client.loop_stop()