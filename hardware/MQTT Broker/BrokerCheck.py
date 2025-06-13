import time
import paho.mqtt.client as mqtt

def on_message(client, userdata, message):
    m = message.payload.decode("utf-8")
    print("Message on "+message.topic+": "+m)

broker_addr = "bd7c7a5e1c8e4513ba43dbbb7f288f38.s1.eu.hivemq.cloud"
broker_port = 8883

client = mqtt.Client(
    client_id = "BROKER_TEST",
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
client.subscribe("#")

while True:
    pass