### LED Driver Software
### ---
### Requests demand data from our web server,
### and then uses a tuned PID controller to
### control power directly.
### ---

led_id = 1

from machine import Pin, I2C, ADC, PWM, reset
from PID import PID
import urequests, network, time, ssl, sys
from umqtt.simple import MQTTClient

# Setup I/O pins
vret_pin = ADC(Pin(26))
vout_pin = ADC(Pin(28))
vin_pin = ADC(Pin(27))
pwm = PWM(Pin(0))
pwm.freq(100000)
pwm_en = Pin(1, Pin.OUT)

# Setup debug LED
led = Pin("LED", Pin.OUT)
led.on()

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
        
r = open("/read.txt", "r")
decision = r.read()
r.close()
if decision == "False":
    flash(led, -1)

f = open("/log.txt", "w")

now = time.gmtime()
f.write("LOG FILE START - {0}:{1}\t{2}\n".format(now[3], now[4], str(now)))
f.flush()
print("LOG FILE START - {0}:{1}\t{2}".format(now[3], now[4], str(now)))

# Setup PID controller
pid = PID(0.05, 10, setpoint=0.3, scale='ms')
#pid.output_limits=[0, P_MAX]

count = 0
pwm_out = 0
pwm_ref = 0
p_request = 0
next_p_request = 0

#flash(led, 1)

try:
    # Setup web connection
    wifi_count = 10
    ssid = "OPPO Find X8"
    password = "18653510219"
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    start_time = time.time_ns()
    wlan.connect(ssid, password)

    f.write("Connecting to WiFi.")
    f.flush()
    print("Connecting to Wifi.",end="")
    while not wlan.isconnected():
        if wifi_count <= 0:
            reset()
        time.sleep(1)
        f.write(".")
        print(".",end="")
        wifi_count -= 1
    time_taken = (time.time_ns() - start_time) * 10**-9
    f.write(" Done! (took {:.3f} seconds)\n".format(time_taken))
    f.write("WiFi connected, IP:" + str(wlan.ifconfig()))
    f.write("\n")
    f.flush()
    print(" Done! (took {:.3f} seconds)".format(time_taken))
    print("WiFi connected, IP:" + str(wlan.ifconfig()))
    print("\n")
    flash(led, 2)

    # Update power setpoint
    def on_message(topic, message):
        topic = topic.decode("utf-8")
        message = message.decode("utf-8")
        print("Message on "+topic+": "+message)
        ima = time.time_ns()
        print(ima)
        global p_request
        global next_p_request
        
        if topic == "sync":
            p_request = next_p_request
        else:
            next_p_request = float(message)

    def saturate(duty):
        if duty > 62500:
            duty = 62500
        if duty < 100:
            duty = 100
        return duty

    broker_addr = b"bd7c7a5e1c8e4513ba43dbbb7f288f38.s1.eu.hivemq.cloud"
    broker_port = 8883
    broker_user = "_LED{0}".format(led_id)
    broker_pwd = b"Password01!"
    client_id = "LED{0}".format(led_id)

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


    start_time = time.time_ns()
    f.write("Connecting to MQTT... ")
    f.flush()
    print("Connecting to MQTT... ",end="")
    client.connect(clean_session=False)

    time_taken = (time.time_ns() - start_time) * 10**-9
    f.write("Done! (took {:.3f} seconds)\n".format(time_taken))
    print("Done! (took {:.3f} seconds)".format(time_taken))

    flash(led, 5)
    
    client.set_callback(on_message)

    f.write("Subscribing to 'LEDs/LED{0}/setpoint'... ".format(led_id))
    f.flush()
    print("Subscribing to 'LEDs/LED{0}/setpoint'... ".format(led_id),end="")
    start_time = time.time_ns()
    client.subscribe("LEDs/LED{0}/setpoint".format(led_id))
    time_taken = (time.time_ns() - start_time) * 10**-9
    f.write(" Done! (took {:.3f} seconds)\n".format(time_taken))
    print(" Done! (took {:.3f} seconds)".format(time_taken))
    flash(led, 3)
    
    f.write("Subscribing to 'sync'... ")
    f.flush()
    print("Subscribing to 'sync'... ",end="")
    start_time = time.time_ns()
    client.subscribe("sync")
    time_taken = (time.time_ns() - start_time) * 10**-9
    f.write(" Done! (took {:.3f} seconds)\n".format(time_taken))
    f.flush()
    print(" Done! (took {:.3f} seconds)".format(time_taken))
    flash(led, 3)
    led.off()
    
except BaseException as e:
    client.disconnect()
    # Prevent log file from being overwritten on restart
    r = open("/read.txt", "w")
    r.write("False")
    r.close()
    
    flash(led, 3, duration=0.5)
    f.write("\n")
    print("ERROR: "+str(e))
    sys.print_exception(e,f)
    f.write("\nSystem Terminated")
    f.flush()
    f.close()
    led.off()
    sys.exit()

state = False 
high = False

# Debugging only
power_log = open("power_log.csv", "w")
power_log.write("POWER LOG FILE START - {0}:{1}\t{2}\n".format(now[3], now[4], str(now)))
power_log.flush()

while True:
    
    client.check_msg()
    pwm_en.value(1)

    vin = 1.026*(12490/2490)*3.3*(vin_pin.read_u16()/65536) # calibration factor * potential divider ratio * ref voltage * digital reading
    vout = 1.026*(12490/2490)*3.3*(vout_pin.read_u16()/65536) # calibration factor * potential divider ratio * ref voltage * digital reading
    vret = 1*3.3*((vret_pin.read_u16()-350)/65536) # calibration factor * potential divider ratio * ref voltage * digital reading
    count = count + 1
    
    # Get new value of p_request
    prev_request = pid.setpoint
    pid.setpoint = min(p_request, 1)
    if prev_request != min(p_request, 1):
        #pid.reset()
        print(str(prev_request)+" -> "+str(p_request))
        state = False
        high = False
    
    # PID control based on power
    p_actual = vout * vret / 1.02
    pwm_ref = pid(p_actual)
    pwm_ref = int(pwm_ref*65536)
    pwm_out = saturate(pwm_ref)
    
    # Stop PID controller after power reaches within 1% of setpoint
    # (avoids oscillations)
    if p_actual > pid.setpoint*0.99 and p_actual < pid.setpoint*1.01:
        #state = True
        if not high:
            high = True
            print("hi")

    if not state:
        pwm.duty_u16(pwm_out)
    
    # Delay 1ms
    time.sleep(1e-3)
    
    # Save power data to CSV, for debugging purposes
    power_log.write(str(p_actual)+","+str(pid.setpoint)+"\n")
    
    if count >= 1000: # After 1000 cycles of PID controller (1 second):
        power_log.flush() # Save power data every second to prevent excessive file write times
        count = 0
        #print("V_out = {:.3f}".format(vout))
        #print("V_ret = {:.3f}".format(vret))
        print("P_request = {:.3f}".format(pid.setpoint))
        print(p_actual)
        print("Duty = {:.3f}".format(pwm_out))
        #print("\n")
        
        # Publish actual power consumption to MQTT broker
        client.publish("LEDs/LED{0}/power".format(led_id), str(p_actual))