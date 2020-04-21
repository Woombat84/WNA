#Control tool for rpi 
import RPi.GPIO as GPIO

#operating system
import os

#control of time 
import time as time

#sensor tools
from goprocam import GoProCamera, constants
from mpu6050 import mpu6050

#tools
import cv2  
from PIL import Image
import numpy as np
import piexif
import math
import urllib

#pin layout
GPIO.setmode(GPIO.BOARD)

#direction pin
dir_pin = 11
GPIO.setup(dir_pin, GPIO.OUT)

#step pin
step_pin = 7
GPIO.setup(step_pin, GPIO.OUT)

#safty pin
safe_pin = 15
GPIO.setup(safe_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

#steplength =  0.1145
steplength =0.0285

#step time per cycle
stepcycle = 0.01

#step time
steptime = stepcycle*0.5

#amount of step for full length of akvrium
step_release = 1000

#this state changes on the interrupt, if the cart has reach the end of the aqurium
state = 1

#holds the time for the complementarry filter for the IMU
dt_comp = time.time()

#alpha is the control nub for the complementary filter
alpha = 0.95

#container for the pitch of in complementary filter
pitch = 0.0

gyro_pitch = 0.0

imu

def save_image(img,_pos,_time,_pitch,number):
    _filename = "{}.jpeg".format("%02d"%number)
    cv2.SaveImage(_filename,img)
    im = Image.open(_filename)
    exif_dict = piexif.load(_filename)
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = "pitch: {} postion: {}".format(_pitch,_pos)
    exif_bytes = piexif.dump(exif_dict)
    im.save(_filename, "jpeg", exif=exif_bytes)
    return

def download_image(photo_url):
    url = urllib.request.urlopen(photo_url)
    photo = np.array(bytearray(url.read()), dtype=np.uint8)
    image = cv2.imdecode(photo, -1)
    return image

def take_image():
    photo_url = goproCamera.take_photo(0)
    return photo_url

def interrupt_handler(channel):
    global state
    if channel == safe_pin:
        if state == 1:
            state = 0
            print("state reset by event on safty pin")


def speed(vel , length):
    counter = 0
    GPIO.output(dir_pin, False) 
    time.sleep(steptime)
    dt = int(1000000000/(vel / steplength))
    t1 = time.clock_gettime_ns()
    while state == 1 and length >  counter * steplength:
        t2 = time.clock_gettime_ns()
        if t2-t1 >= dt:
            
            GPIO.output(step_pin, True)
            time.sleep(steptime)
            GPIO.output(step_pin, False)
            time.sleep(steptime)
            counter = counter +1
            imu_data()
    step_release = counter
    print(counter)
    return counter * steplength
    

def imu_data():
     global dt_comp, pitch, alpha,gyro_pitch ,imu
     dt = time.time() - dt_comp
     dt_comp = time.time()
     try:
        imu_accel, imu_gyro, imu_temp = imu.get_all_data()
        accel_pitch = math.degrees(math.atan2(imu_accel['z'],imu_accel['y']))
        gyro_pitch = gyro_pitch + imu_gyro['x'] * dt/1000
        pitch = alpha * (pitch + gyro_pitch) + (1 - alpha) * accel_pitch
     except:
        print('An error occured.')
    
     return pitch
 

def setup_int():
    
    #setup of safety interupt
    GPIO.add_event_detect(safe_pin, GPIO.FALLING,
                      callback=interrupt_handler,
                      bouncetime=200)
    

    return
def setup_imu():
    
    
    #setup of imu
    global imu 
    #address of imu is 68
    imu = mpu6050(0x68)
    imu.set_accel_range(ACCEL_RANGE_2G)
    imu.set_gyro_range(GYRO_RANGE_250DEG)
    

    return
def setup_gopro():
    
    global  goproCamera
    #initiating object for go pro camera
    goproCamera = GoProCamera.GoPro()
    #camera settings adjusted
    goproCamera.gpControlSet(constants.Photo.RESOLUTION,
                            constants.Photo.Resolution.R12W)
    goproCamera.locate(constants.Locate.Start)
    return
# holdes the data aquriede to easy storing per image
class image_holder(obj):
    
    def __init__(self,path,pos,pitch,time,number):
        self.path = path
        self.pos = pos 
        self.pitch = pitch
        self.time = time
        self.number = number



def clean():
    GPIO.cleanup()
    exit(0)