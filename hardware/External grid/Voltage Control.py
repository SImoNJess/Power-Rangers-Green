from machine import Pin, I2C, ADC, PWM, Timer
import network, ssl
from umqtt.simple import MQTTClient
import utime

# Save Vb and power for test results
SAVE_RESULTS = False
RESULTS_FILE = "results.csv"

# WiFi Setup
#ssid = "HotspotTest"
#password = "passtest"

ssid = "OPPO Find X8"
password = "18653510219"

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(ssid, password)
print("Connecting to WiFi...", end="")
while not wlan.isconnected():
    utime.sleep(0.5)
    print(".", end="")
print(" Done!\nWiFi connected, IP:", wlan.ifconfig())

broker_addr = b"bd7c7a5e1c8e4513ba43dbbb7f288f38.s1.eu.hivemq.cloud"
broker_port = 8883
broker_user = b"ExtGrid"
broker_pwd = b"Password01!"
client_id = b"ExtGrid"

context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
context.verify_mode = ssl.CERT_NONE

client = MQTTClient(
    client_id = client_id,
    server = broker_addr,
    port = broker_port,
    user = broker_user,
    password = broker_pwd,
    keepalive = 30,
    ssl = context
)

client.DEBUG = True

print("Connecting to MQTT... ")
client.connect(clean_session=False)

# Set up some pin allocations for the Analogues and switches
va_pin = ADC(Pin(28))
vb_pin = ADC(Pin(26))

# Set up the I2C for the INA219 chip for current sensing
ina_i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=2400000)

# Some PWM settings, pin number, frequency, duty cycle limits and start with the PWM outputting the default of the min value.
pwm = PWM(Pin(9))
pwm.freq(100000)
min_pwm = 1000
max_pwm = 64536
pwm_out = min_pwm
pwm_ref = 30000

kp = 150 # Boost Proportional Gain
ki = 300 # Boost Integral Gain

v_ref = 7.05 # Voltage reference for the CL modes
v_err = 0 # Voltage error
v_err_int = 0 # Voltage error integral
v_pi_out = 0 # Output of the voltage PI controller

# Basic signals to control logic flow
global timer_elapsed
timer_elapsed = 0
count = 0
first_run = 1

# Need to know the shunt resistance
global SHUNT_OHMS
SHUNT_OHMS = 0.10

state = False
high = False

# saturation function for anything you want saturated within bounds
def saturate(signal, upper, lower): 
    if signal > upper:
        signal = upper
    if signal < lower:
        signal = lower
    return signal

# This is the function executed by the loop timer, it simply sets a flag which is used to control the main loop
def tick(t): 
    global timer_elapsed
    timer_elapsed = 1



# These functions relate to the configuring of and reading data from the INA219 Current sensor
class ina219: 
    
    # Register Locations
    REG_CONFIG = 0x00
    REG_SHUNTVOLTAGE = 0x01
    REG_BUSVOLTAGE = 0x02
    REG_POWER = 0x03
    REG_CURRENT = 0x04
    REG_CALIBRATION = 0x05
    
    def __init__(self,sr, address, maxi):
        self.address = address
        self.shunt = sr
            
    def vshunt(icur):
        # Read Shunt register 1, 2 bytes
        reg_bytes = ina_i2c.readfrom_mem(icur.address, icur.REG_SHUNTVOLTAGE, 2)
        reg_value = int.from_bytes(reg_bytes, 'big')
        if reg_value > 2**15: #negative
            sign = -1
            for i in range(16): 
                reg_value = (reg_value ^ (1 << i))
        else:
            sign = 1
        return (float(reg_value) * 1e-5 * sign)
        
    def vbus(ivolt):
        # Read Vbus voltage
        reg_bytes = ina_i2c.readfrom_mem(ivolt.address, ivolt.REG_BUSVOLTAGE, 2)
        reg_value = int.from_bytes(reg_bytes, 'big') >> 3
        return float(reg_value) * 0.004
        
    def configure(conf):
        #ina_i2c.writeto_mem(conf.address, conf.REG_CONFIG, b'\x01\x9F') # PG = 1
        #ina_i2c.writeto_mem(conf.address, conf.REG_CONFIG, b'\x09\x9F') # PG = /2
        ina_i2c.writeto_mem(conf.address, conf.REG_CONFIG, b'\x19\x9F') # PG = /8
        ina_i2c.writeto_mem(conf.address, conf.REG_CALIBRATION, b'\x00\x00')


if SAVE_RESULTS:
    results = open(RESULTS_FILE, "w")

# Here we go, main function, always executes
while True:    
    if first_run:
        # for first run, set up the INA link and the loop timer settings
        ina = ina219(SHUNT_OHMS, 64, 5)
        ina.configure()
        first_run = 0
        
        # This starts a 1kHz timer which we use to control the execution of the control loops and sampling
        loop_timer = Timer(mode=Timer.PERIODIC, freq=1000, callback=tick)
    
    # If the timer has elapsed it will execute some functions, otherwise it skips everything and repeats until the timer elapses
    if timer_elapsed == 1: # This is executed at 1kHz
        va = 1.017*(12490/2490)*3.3*(va_pin.read_u16()/65536) # calibration factor * potential divider ratio * ref voltage * digital reading
        vb = 1.015*(12490/2490)*3.3*(vb_pin.read_u16()/65536) # calibration factor * potential divider ratio * ref voltage * digital reading
        
        Vshunt = ina.vshunt()
        
        # New min and max PWM limits and we use the measured current directly
        min_pwm = 0 
        max_pwm = 64536
        iL = Vshunt/SHUNT_OHMS
        
        power = vb * iL


        # PID Controller
        v_err = v_ref - vb # calculate the error in power
        v_err_int += v_err # add it to the integral error
        v_err_int = saturate(v_err_int, 10000, -10000) # saturate the integral error
        v_pi_out = (kp * v_err) + (ki * v_err_int) # Calculate a PI controller output    
        pwm_out = saturate(v_pi_out, max_pwm, min_pwm) # Saturate that PI output
        duty = int(65536-pwm_out) # Invert because reasons
        pwm.duty_u16(duty) # Send the output of the PI controller out as PWM
     
        # Keep a count of how many times we have executed and reset the timer so we can go back to waiting
        count = count + 1
        timer_elapsed = 0
        
        # Test result gathering
        if SAVE_RESULTS:
            results.write(f"{vb:.3f},{power:.3f},{duty}\n")
        
        if count > 1000:
            if SAVE_RESULTS:
                results.flush()
            if power < 0:
                export_power = -power
                import_power = 0
            else:
                export_power = 0
                import_power = power
                
            client.publish("External/export_power", f"{export_power:.2f}")
            client.publish("External/import_power", f"{import_power:.2f}")
            count = 0
            print("Vb = {:.3f}".format(vb))
            print("Power = {:.3f}".format(power))
            print("duty cycle = {:.3f}".format(duty))

        