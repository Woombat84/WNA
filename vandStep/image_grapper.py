#Control tool for rpi 
import RPi.GPIO as GPIO

#operating system
import os

#control of time 
import time as time
from datetime import date
import datetime
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
dir_pin = 7
GPIO.setup(dir_pin, GPIO.OUT)

#step pin
step_pin = 11
GPIO.setup(step_pin, GPIO.OUT)

#safty pin
safe_pin = 15
GPIO.setup(safe_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

#steplength in mm
steplength = 0.0285

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
imu_angle = 0.0

#date of capturing
today = date.today()
cap_date = today.strftime("%m%d%Y")



def save_image(img,_pos,_time,_pitch,number):
    _filename = "{}.jpeg".format("%03d"%number)
    cv2.imwrite(_filename,img)
    im = Image.open(_filename)
    exif_dict = piexif.load(_filename)
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = "pitch: {} postion: {}".format(_pitch,_pos)
    exif_bytes = piexif.dump(exif_dict)
    im.save(_filename, "jpeg", exif=exif_bytes)
    im.close()
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


def speed(length):
    global pitch, imu_angle
    counter = 0
    GPIO.output(dir_pin, False) 
    time.sleep(steptime)
    while state == 1 and length >  counter * steplength:
        counter = counter +1
        GPIO.output(step_pin, True)
        time.sleep(steptime)
        GPIO.output(step_pin, False)
        time.sleep(steptime)    
    step_release = counter
    #print(counter)
    return counter * steplength , imu_angle
    

def imu_data():
     global dt_comp, pitch, alpha,gyro_pitch,imu
     try:
        dt = time.time() - dt_comp
        dt_comp = time.time()
        imu_accel, imu_gyro, imu_temp = imu.get_all_data()
        accel_pitch = math.degrees(math.atan2(imu_accel['z'],imu_accel['y']))
        gyro_pitch = gyro_pitch + imu_gyro['x'] * dt/1000.0
        pitch = alpha * (pitch + gyro_pitch) + (1 - alpha) * accel_pitch
     except:
        print("erroer")
     

     return int(pitch)

def setup():
    ##setup of safety interupt / stop pin
    global imu_angle
    GPIO.add_event_detect(safe_pin, GPIO.FALLING,
                      callback=interrupt_handler,
                      bouncetime=200)
    ##setup of imu
    global imu 
    #address of imu is 68
    imu = mpu6050(0x68)
    imu.set_accel_range(0)
    imu.set_gyro_range(0)
    #calculating pitch has to be estimated before hand
    for i in range(100):
        imu_angle = imu_data()

    ##setup of camera
    global  goproCamera
    #initiating object for go pro camera
    goproCamera = GoProCamera.GoPro()
    #camera mode
    goproCamera.mode(constants.Mode.PhotoMode, constants.Mode.SubMode.Photo.Single)
    
    #camera setup
    
    #Orientation
    goproCamera.gpControlSet(constants.Setup.ORIENTATION,
                            constants.Photo.Orientation.Auto)
    #turning off display
    goproCamera.gpControlSet(constants.Setup.DISPLAY,
                            constants.Photo.Display.OFF)
    #photo setup

    #camera settings adjusted -- R12L,R12W
    goproCamera.gpControlSet(constants.Photo.RESOLUTION,
                            constants.Photo.Resolution.R12L)
    #HDR control
    goproCamera.gpControlSet(constants.Photo.HDR_PHOTO,
                            constants.Photo.HDR.ON)
    #supershot 
    goproCamera.gpControlSet(constants.Photo.SUPER_PHOTO,
                            constants.Photo.SuperPhoto.Auto)
    #sharpness
    goproCamera.gpControlSet(constants.Photo.SHARPNESS,
                            constants.Photo.Sharpness.High)

    #timer
    goproCamera.gpControlSet(constants.Photo.PHOTO_TIMER,
                            constants.Photo.PhotoTimer.OFF)




    
    #libary change and creation
    now = datetime.datetime.now()
    clock = "{}{}".format(now.hour,"%02d"%now.minute)
    os.chdir('/home/pi/project/pictures')
    _dir = "{}_{}_{}".format(cap_date,clock,imu_angle)
    os.mkdir(_dir)
    os.chdir(_dir)
    
    return



# holdes the data aquriede to easy storing per image
class image_holder():
    
    def __init__(self,path,pos,pitch,time,number):
        self.path = path
        self.pos = pos 
        self.pitch = pitch
        self.time = time
        self.number = number

def main():
    setup()
    image_list = []
    photo_count = 0
    i_pos = 0
    while state == 1:
       pos,i_pitch = speed(1.0)
       i_pos = pos + i_pos
       try:
            i_path = take_image()
            i_time = time.localtime()
            image_obj = image_holder(i_path,i_pos,i_pitch,i_time,photo_count)
            image_list.append(image_obj) 
            photo_count = photo_count + 1
       except:
            print("missed an image")

    print("image cap done")
    for obj in image_list:
        print("number {} out of {}".format(obj.number+1,photo_count))
        image = download_image(obj.path)
        save_image(image,obj.pos,obj.time,obj.pitch,obj.number)

    print('done complet')
    GPIO.cleanup()
    exit(1)

if __name__ == "__main__":
    main()


