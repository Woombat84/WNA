import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)

channel = 13

#direction pin
GPIO.setup(11, GPIO.OUT)
#step pin
GPIO.setup(7, GPIO.OUT)
#safty pin
GPIO.setup(channel, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
bB = True
while bB:
    if GPIO.input(channel):
        print('Input was HIGH')
    else:
        print('Input was LOW')
        bB = False

        
GPIO.cleanup()
