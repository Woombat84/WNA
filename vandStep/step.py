import RPi.GPIO as GPIO
import time as time

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
steplength =0.1145

#step time per cycle
stepcycle = 0.001

#step time
steptime = stepcycle*0.5

#amount of step for full length of akvrium
step_release = 1000

state = 1


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
    GPIO.output(dir_pin, False) 
    time.sleep(steptime)
    while counter > 100:
       GPIO.output(step_pin, True) 
       time.sleep(steptime)
       GPIO.output(step_pin, False) 
       time.sleep(steptime)
       counter = counter + 1
        
    print("done test : " + counter)
    
    return


def main():
    GPIO.add_event_detect(safe_pin, GPIO.FALLING,
                      callback=interrupt_handler,
                      bouncetime=200)
    test()

if __name__ == "__main__":
    main()

GPIO.cleanup()
