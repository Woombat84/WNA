#Control tool for rpi 
import RPi.GPIO as GPIO

#operating system
import os
import sys
import subprocess

#control of time 
import time as time
from datetime import date
import datetime
#sensor tools
from goprocam import GoProCamera, constants
from mpu6050 import mpu6050

#tools

import urllib

#pin layout
GPIO.setmode(GPIO.BOARD)

#direction pin
dir_pin = 7
GPIO.setup(dir_pin, GPIO.OUT)

#step pin
step_pin = 11
GPIO.setup(step_pin, GPIO.OUT)

#led finsih pin
finish_led = 37
GPIO.setup(finish_led, GPIO.OUT)
#safty pin
safe_pin = 15
GPIO.setup(safe_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

#steplength in mm
steplength = 0.0285

#step time per cycle
stepcycle = 0.030

#step time
steptime = stepcycle*0.5

#amount of step for full length of akvrium
step_release = 0

#this state changes on the interrupt, if the cart has reach the end of the aqurium
state = 1

#holds the time for the complementarry filter for the IMU
dt_comp = time.time()

#alpha is the control nub for the complementary filter
alpha = 0.95

#container for the pitch of in complementary filter
pitch = 0.0
gyro_pitch = 0.0
imu_angle = 0.0

#date of capturing
today = date.today()
cap_date = today.strftime("%m%d%Y")


#input arguments for the weed number as annotation for the image group
_args =""
def imu_data():
     global dt_comp, pitch, alpha,gyro_pitch,imu
     _err ="no"
     try:
        dt = time.time() - dt_comp
        dt_comp = time.time()
        imu_accel, imu_gyro, imu_temp = imu.get_all_data()
        accel_pitch = math.degrees(math.atan2(imu_accel['z'],imu_accel['y']))
        gyro_pitch = gyro_pitch + imu_gyro['x'] * dt/1000.0
        pitch = alpha * (pitch + gyro_pitch) + (1 - alpha) * accel_pitch
        return _err, int(pitch)
     except:
        print("pitch erroer")
        _err = "a"
        return _err, int(pitch)
     

def camera_control():

    return

def interrupt_handler(channel):
    global state

    print("interrupt handler")

    if channel == safe_pin:
        if state == 1:
            state = 0
            print("state reset by event on pin 13")

def release():
    relax_time = 1.0*0.5
    #clockwise rotation
    GPIO.output(dir_pin, True) 
    time.sleep(0.05)
    for x in range (0,step_release-1):
        GPIO.output(step_pin, True) 
        time.sleep(steptime*10)
        GPIO.output(step_pin, False) 
        time.sleep(steptime*10)
    return


def speed(vel):
    counter = 0
    GPIO.output(dir_pin, False) 
    time.sleep(steptime)
    dt = int(1000000000/(vel / steplength))
    t1 = time.clock_gettime_ns()
    while GPIO.input(safe_pin) == GPIO.HIGH:
        t2 = time.clock_gettime_ns()
        if t2-t1 >= dt:
            counter = counter +1
            GPIO.output(step_pin, True) 
            time.sleep(steptime)
            GPIO.output(step_pin, False) 
            time.sleep(steptime)
    step_release = counter
    print(counter)
    return

  
def test():
    #counter clockwise rotation
    counter = 0
    run = 50000
    GPIO.output(dir_pin, False) 
    time.sleep(steptime)
    while counter < run and state == 1:
       GPIO.output(step_pin, True) 
       time.sleep(steptime)
       GPIO.output(step_pin, False) 
       time.sleep(steptime)
       counter = counter + 1
        
    print("done test : {}".format(counter))
    
    return
def setup():

    global _args
    _args = int(sys.argv[1])
    if not _args:
        _args = input("please add a weed number: ")

    ##setup of safety interupt / stop pin
    global imu_angle
    GPIO.add_event_detect(safe_pin, GPIO.FALLING,
                      callback=interrupt_handler,
                      bouncetime=200)
    print("setup of interrupt done")
    '''
    ##setup of imu
    #global imu 
   #address of imu is 68
    #imu = mpu6050(0x68)
    #imu.set_accel_range(0)
    #imu.set_gyro_range(0)
    #calculating pitch has to be estimated before hand
    #n = 100
    #for i in range(n):
    #    err , imu_angle = imu_data()
    #    while err == "a":
    #        n = n + 1
    #       err, imu_angle = imu_data()
            
    #print("setup of imu, and angle is: {}".format(imu_angle))
    '''
  
    #libary change and creation
    now = datetime.datetime.now()
    clock = "{}{}".format(now.hour,"%02d"%now.minute)
    os.chdir('/home/pi/project/pictures')
    _dir = "{}_{}_{}".format(_args,cap_date,clock)
    os.mkdir(_dir)
    os.chdir(_dir)
    
    return

def main():
    
    
    setup()
    test()

if __name__ == "__main__":
    main()

GPIO.cleanup()
