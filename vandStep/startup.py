import RPi.GPIO as GPIO
import time as time



#pin layout
GPIO.setmode(GPIO.BOARD)

#on off pin
on_off_pin = 16
GPIO.setup(on_off_pin, GPIO.OUT)

#power pin
power_pin = 18
GPIO.setup(power_pin, GPIO.OUT,initial=GPIO.HIGH)


time.sleep(0.1)

while 1:
    print('start time')
    if time.localtime().tm_hour >= 8 and time.localtime().tm_hour < 17:
        GPIO.output(on_off_pin,GPIO.LOW)
        time.sleep(3600)
        
    else:
        GPIO.output(on_off_pin,GPIO.HIGH)
        time.sleep(3600)

        
    time.sleep(3)


