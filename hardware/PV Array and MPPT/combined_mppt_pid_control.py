from machine import Pin, I2C, ADC, PWM, Timer
import network, ssl
from umqtt.simple import MQTTClient
import utime, time

# Pins and Interfaces
led = Pin("LED", Pin.OUT)
va_pin = ADC(Pin(28))
vb_pin = ADC(Pin(26))
ina_i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=2400000)
pwm = PWM(Pin(9))
pwm.freq(100000)

# Constants
SHUNT_OHMS = 0.10
min_pwm = 1000
max_pwm = 40000
pwm_out = 12300
pwm_step = 100
MPP = 3.2  # Max PV power
P_desired = 1.5  # Initial power target
command = 0  # Default to PID control

# WiFi Setup
ssid = "OPPO Find X8"
password = "18653510219"
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(ssid, password)
while not wlan.isconnected():
    utime.sleep(0.5)
print("WiFi connected:", wlan.ifconfig())

# MQTT Setup
def on_message(topic, msg):
    global command, P_desired
    topic = topic.decode()
    msg = msg.decode()
    if topic == "PV/mode":
        try:
            command = int(msg)
            print(f"[MQTT] Mode changed to: {command}")
        except:
            print("[MQTT] Invalid mode received")
    elif topic == "PV/irradiance":
        try:
            irr = max(0, min(100, int(msg)))
            P_desired = round((irr * MPP) / 100, 2)
            print(f"[MQTT] Irradiance: {irr}%, P_desired = {P_desired:.2f} W")
        except:
            print("[MQTT] Invalid irradiance value")

client = MQTTClient(
    client_id=b"PVController",
    server=b"bd7c7a5e1c8e4513ba43dbbb7f288f38.s1.eu.hivemq.cloud",
    port=8883,
    user=b"server1",
    password=b"Password01!",
    keepalive=7200,
    ssl=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
)
client.set_callback(on_message)
client.connect()
client.subscribe("PV/irradiance")
client.subscribe("PV/mode")
print("MQTT connected and subscribed.")

# INA219 driver
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

ina = ina219(SHUNT_OHMS, 64)
ina.configure()

# PID Variables
kp, ki, kd = 150, 5, 10
v_err_int, previous_v_err = 0, 0
integral_min, integral_max = -5000, 5000
power_sum, sample_count, pid_count = 0, 0, 0

# MPPT Variables
prev_power = 0
pwm_direction = 1
mppt_count = 0

# Timer
timer_elapsed = 0
def tick(t): global timer_elapsed; timer_elapsed = 1
loop_timer = Timer(mode=Timer.PERIODIC, freq=1000, callback=tick)

# Main Loop
while True:
    client.check_msg()
    if timer_elapsed:
        timer_elapsed = 0
        va = 1.017 * (12490 / 2490) * 3.3 * (va_pin.read_u16() / 65536)
        vb = 1.015 * (12490 / 2490) * 3.3 * (vb_pin.read_u16() / 65536)
        iL = -ina.vshunt() / SHUNT_OHMS
        power = vb * iL

        if command == 1:
            # MPPT Mode
            led.off()
            Ppv = round(power, 3)

            if Ppv > prev_power:
                pwm_out += pwm_step * pwm_direction
            else:
                pwm_direction *= -1
                pwm_out += pwm_step * pwm_direction
            pwm_out = max(min(pwm_out, max_pwm), min_pwm)
            pwm.duty_u16(pwm_out)
            prev_power = Ppv
            utime.sleep_ms(10)

            mppt_count += 1
            if mppt_count >= 200:
                print("MPPT → Vpv: {:.2f} V, Ipv: {:.2f} A, Ppv: {:.2f} W".format(vb, iL, Ppv))
                client.publish("PV/power", f"{Ppv:.2f}")
                client.publish("PV/vbus", f"{va:.2f}")
                mppt_count = 0
 
        else:
            # PID Power Control Mode
            led.on()
            error = P_desired - power
            v_err_int = max(min(v_err_int + error, integral_max), integral_min)
            v_err_deriv = error - previous_v_err
            pid = kp * error + ki * v_err_int + kd * v_err_deriv
            previous_v_err = error

            pwm_out = max(min(pwm_out + int(pid), max_pwm), min_pwm)
            pwm.duty_u16(pwm_out)

            power_sum += power
            sample_count += 1
            pid_count += 1
            if pid_count >= 5000:
                avg_power = power_sum / sample_count
                client.publish("PV/power", f"{avg_power:.2f}")
                client.publish("PV/vbus", f"{va:.2f}")
                print(f"PID → P = {avg_power:.2f} W, Vbus = {va:.2f} V")
                power_sum = 0
                sample_count = 0
                pid_count = 0
