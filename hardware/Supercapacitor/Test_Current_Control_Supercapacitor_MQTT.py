# Current Control for Supercapacitor Energy Management
# Performs: Store 5J (S), Extract 5J (E), and Maintain (U)
# Uses PID to control current instead of power to improve startup stability

from machine import Pin, I2C, ADC, PWM, Timer
import time, network, ssl
from umqtt.simple import MQTTClient

# PWM Setup
va_pin = ADC(Pin(28))
ina_i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=2400000)
pwm = PWM(Pin(9))
pwm.freq(100000)
min_pwm = 1000
max_pwm = 64536
duty = min_pwm
pwm.duty_u16(duty)

# Setup debug LED
led = Pin("LED", Pin.OUT)
led.on()

# LED flash subroutine
def flash(led, n, duration=0.1):
    if n == -1:
        while True:
            led.off()
            time.sleep(duration)
            led.on()
            time.sleep(duration)
    if led.value() == 1:
        for i in range(n):
            led.off()
            time.sleep(duration)
            led.on()
            time.sleep(duration)
    else:
        for i in range(n):
            led.on()
            time.sleep(duration)
            led.off()
            time.sleep(duration)

# WiFi Setup
ssid = "OPPO Find X8"
password = "18653510219"
wlan = network.WLAN(network.STA_IF)
wlan.active(True)

start_time = time.time_ns()
wlan.connect(ssid, password)

print("Connecting to WiFi...", end="")
while not wlan.isconnected():
    time.sleep(0.5)
    print(".", end="")
time_taken = (time.time_ns() - start_time) * 10**-9
print(f" Done! (took {time_taken:.3f} seconds)\nIP:", wlan.ifconfig())
flash(led, 2)

# MQTT Setup
command = "U"  # Default command
def on_message(topic, msg):
    global command
    try:
        command = msg.decode().strip().upper()
        print("--- Received MQTT command:", command,"---")
        if command not in ("S", "E", "H"):
            command = "U"
    except:
        command = "U"

broker_addr = b"bd7c7a5e1c8e4513ba43dbbb7f288f38.s1.eu.hivemq.cloud"
broker_port = 8883
broker_user = b"server1"
broker_pwd = b"Password01!"
client_id = b"SuperCap"

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

start_time = time.time_ns()
print("Connecting to MQTT... ", end="")
client.connect()
time_taken = (time.time_ns() - start_time) * 10**-9
print(f"Done! (took {time_taken:.3f} seconds)")
flash(led, 5)

start_time = time.time_ns()
print("Subscribing to 'capacitor/cmd'... ", end="")
client.subscribe("capacitor/cmd")
time_taken = (time.time_ns() - start_time) * 10**-9
print(f"Done! (took {time_taken:.3f} seconds)")
flash(led, 3)
led.off()

C = 0.25  # Farads
SHUNT_OHMS = 0.10
dt = 1 / 1000.0  # 1 kHz sampling
max_capacity = C * 14.5**2
min_capacity = C * 8**2

timer_elapsed = 0
count = 0
first_run = 1

# PID gains for current control
kp, ki, kd = 10, 0.05, 0
int_err = 0
prev_err = 0
integral_min = -5000
integral_max = 5000

I_target = 0.0
E_delta = 0.0
action_in_progress = False

step_ticks = 5000  # 5 sec
E_target = 5.0

min_reached = False
max_reached = False

class ina219:
    REG_SHUNTVOLTAGE = 0x01
    def __init__(self, shunt, address):
        self.address = address
        self.shunt = shunt

    def vshunt(self):
        reg_bytes = ina_i2c.readfrom_mem(self.address, self.REG_SHUNTVOLTAGE, 2)
        reg_value = int.from_bytes(reg_bytes, 'big')
        if reg_value > 2**15:
            reg_value -= 2**16
        return float(reg_value) * 1e-5

    def configure(self):
        ina_i2c.writeto_mem(self.address, 0x00, b'\x19\x9F')
        ina_i2c.writeto_mem(self.address, 0x05, b'\x00\x00')

def tick(t):
    global timer_elapsed
    timer_elapsed = 1

def saturate(val, high, low):
    return max(min(val, high), low)

while True:
    if first_run:
        ina = ina219(SHUNT_OHMS, 64)
        ina.configure()
        loop_timer = Timer(mode=Timer.PERIODIC, freq=1000, callback=tick)
        first_run = 0

    if timer_elapsed == 1:
        timer_elapsed = 0

        va = 1.017 * (12490 / 2490) * 3.3 * (va_pin.read_u16() / 65536)
        Vshunt = ina.vshunt()
        IL = -Vshunt / SHUNT_OHMS
        E_stored = C * va**2
        power_output = va * IL
        SoC = E_stored / 50 *100
        if SoC < 0:
            SoC = 0
        elif SoC > 100:
            SoC = 100

            
        if E_stored < min_capacity and not min_reached:
            print(f"Energy reached minimum! E_stored = {E_stored:.3f} < {min_capacity:.3f}")
            action_in_progress = False
            command = "U"
            min_reached = True
            I_target = 0.1
        
        if E_stored > (min_capacity+1) and min_reached:
            print(f"Energy is no longer at minimum, E_stored = {E_stored:.3f}")
            I_target = 0.02
            min_reached = False
        
        if E_stored > max_capacity and not max_reached:
            print(f"Energy reached maximum! E_stored = {E_stored:.3f} > {max_capacity:.3f}")
            action_in_progress = False
            command = "U"
            max_reached = True
            I_target = -0.1
        
        if E_stored < (max_capacity-1) and max_reached:
            print(f"Energy is no longer at maximum, E_stored = {E_stored:.3f}")
            I_target = 0.02
            max_reached = False

        client.check_msg()

        if command != "U" and not min_reached and not max_reached and not action_in_progress: # New command has been received
            E_delta = 0.0
            E_initial = E_stored
            if command == "S":
                I_target = 0.2
                action_in_progress = True
                #print("--- ACTION: STORE 5J ---")
            elif command == "E":
                I_target = -0.2
                action_in_progress = True
                #print("--- ACTION: EXTRACT 5J ---")
            #elif command == "H":
            else:
                I_target = 0.02
                action_in_progress = False
                #print("--- ACTION: MAINTAIN ENERGY ---")

        # Stop current injection when energy delta goal is met
        if action_in_progress:
            E_delta = E_stored - E_initial
            if abs(E_delta) >= E_target:
                I_target = 0.02
                action_in_progress = False
                command = "U"

        # PID Current Control
        err = I_target - IL
        int_err += err
        int_err = saturate(int_err, integral_max, integral_min)
        der_err = err - prev_err

        pid_output = kp * err + ki * int_err + kd * der_err
        prev_err = err

        duty += int(pid_output)
        duty = saturate(duty, max_pwm, min_pwm)
        pwm.duty_u16(duty)

        if count % 200 == 0:
            print(command, "min_reached = {0}, max_reached = {1}".format(min_reached, max_reached))
            print(f"I = {IL*1000:.1f} mA, I_target = {I_target:.3f} A, V = {va:.2f} V, P = {power_output:.2f} W, E = {E_stored:.2f} J, ΔE = {E_delta:.2f}, Duty = {duty}")
            print(SoC)
            client.publish("capacitor/energy", f"{E_stored:.2f}")
            client.publish("capacitor/SoC", f"{SoC:.2f}")
            client.publish("capacitor/E_delta", f"{E_delta:.2f}")
        count += 1