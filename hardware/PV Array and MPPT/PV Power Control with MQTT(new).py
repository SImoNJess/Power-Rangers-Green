from machine import Pin, I2C, ADC, PWM, Timer
import network, ssl
from umqtt.simple import MQTTClient
import utime

# Blink onboard LED at boot
led = Pin("LED", Pin.OUT)  # Use "LED" for built-in LED on Pico W
for _ in range(3):
    led.on()
    time.sleep(0.2)
    led.off()
    time.sleep(0.2)
print("Boot LED blink complete.")

# Hardware Setup
va_pin = ADC(Pin(28))
vb_pin = ADC(Pin(26))
ina_i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=2400000)
pwm = PWM(Pin(9))
pwm.freq(100000)

min_pwm = 1000
max_pwm = 40000
pwm_out = 1000
pwm.duty_u16(pwm_out)

# WiFi Setup
ssid = "HotspotTest"
password = "passtest"
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(ssid, password)
print("Connecting to WiFi...", end="")
while not wlan.isconnected():
    utime.sleep(0.5)
    print(".", end="")
print(" Done!\nWiFi connected, IP:", wlan.ifconfig())

# MQTT Setup
def on_message(topic, msg):
    global P_desired
    topic = topic.decode()
    msg = msg.decode()
    if topic == "PV/irradiance":
        try:
            irr = max(0, min(100, int(msg)))  # Clamp to [0,100]
            P_desired = round((irr * MPP) / 100, 2)
            print(f"Received Irradiance: {irr}%, P_desired updated to: {P_desired:.2f} W")
        except:
            print("Invalid MQTT irradiance value received.")

broker_addr = b"bd7c7a5e1c8e4513ba43dbbb7f288f38.s1.eu.hivemq.cloud"
broker_port = 8883
broker_user = b"server1"
broker_pwd = b"Password01!"
client_id = b"PVController"

context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
context.verify_mode = ssl.CERT_NONE

client = MQTTClient(
    client_id=client_id,
    server=broker_addr,
    port=broker_port,
    user=broker_user,
    password=broker_pwd,
    keepalive=7200,
    ssl=context
)

client.set_callback(on_message)
client.connect()
client.subscribe("PV/irradiance")
print("MQTT connected and subscribed to PV/irradiance")

SHUNT_OHMS = 0.10
MPP = 3.2  # Max PV power at 100% irradiance
P_desired = 1.5  # Initial default

# PID Parameters
kp, ki, kd = 150, 5, 10
v_err_int, previous_v_err = 0, 0
integral_min, integral_max = -5000, 5000

# INA219 Driver
class ina219:
    REG_CONFIG = 0x00
    REG_SHUNTVOLTAGE = 0x01
    REG_CALIBRATION = 0x05
    def __init__(self, shunt, address):
        self.address = address
        self.shunt = shunt
    def vshunt(self):
        val = int.from_bytes(ina_i2c.readfrom_mem(self.address, self.REG_SHUNTVOLTAGE, 2), 'big')
        if val > 0x7FFF: val -= 0x10000
        return val * 1e-5
    def configure(self):
        ina_i2c.writeto_mem(self.address, self.REG_CONFIG, b'\x19\x9F')
        ina_i2c.writeto_mem(self.address, self.REG_CALIBRATION, b'\x00\x00')

# Timer Setup
timer_elapsed = 0
def tick(t):
    global timer_elapsed
    timer_elapsed = 1

# Run Loop
ina = ina219(SHUNT_OHMS, 64)
ina.configure()
loop_timer = Timer(mode=Timer.PERIODIC, freq=1000, callback=tick)

power_sum = 0
sample_count = 0
count = 0

while True:
    global v_err_int, previous_v_err
    client.check_msg()
    if timer_elapsed:
        timer_elapsed = 0

        # Read voltage and current
        va = 1.017 * (12490 / 2490) * 3.3 *(va_pin.read_u16() / 65536)
        vb = 1.015 * (12490 / 2490) * 3.3 * (vb_pin.read_u16() / 65536)
        iL = -ina.vshunt() / SHUNT_OHMS
        power = vb * iL

        # PID control
        error = P_desired - power
        v_err_int = max(min(v_err_int + error, integral_max), integral_min)
        v_err_deriv = error - previous_v_err
        pid = kp * error + ki * v_err_int + kd * v_err_deriv
        previous_v_err = error

        pwm_out = max(min(pwm_out + int(pid), max_pwm), min_pwm)
        pwm.duty_u16(pwm_out)

        # Track for average power
        power_sum += power
        sample_count += 1
        count += 1

        # Every 5s, send average power
        if count >= 5000:
            avg_power = power_sum / sample_count
            client.publish("PV/power", f"{avg_power:.2f}")
            client.publish("PV/vbus", f"{va:.2f}")
            power_sum = 0
            sample_count = 0
            count = 0
