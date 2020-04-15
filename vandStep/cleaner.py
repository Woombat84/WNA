import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)

#direction pin
GPIO.setup(11, GPIO.OUT)
#step pin
GPIO.setup(7, GPIO.OUT)
#safty pin
GPIO.setup(13, GPIO.IN)

GPIO.remove_event_detect(13)
GPIO.cleanup()
