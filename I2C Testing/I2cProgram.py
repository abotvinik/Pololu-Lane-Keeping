import RPi.GPIO as GPIO
import time

# Set the GPIO mode to BCM
GPIO.setmode(GPIO.BCM)

# Define the GPIO pin number
gpio_pin = 17  # You can use a different GPIO pin number if needed

# Set up the GPIO pin as an output
GPIO.setup(gpio_pin, GPIO.OUT)

try:
    # Print a message
    print(f"Toggle GPIO pin {gpio_pin} every second. Press Ctrl+C to exit.")

    # Keep toggling the GPIO pin
    while True:
        GPIO.output(gpio_pin, GPIO.HIGH)  # Turn to high
        time.sleep(1)
        
        GPIO.output(gpio_pin, GPIO.LOW)   # Turn to low
        time.sleep(1)

except KeyboardInterrupt:
    # Handle keyboard interrupt (Ctrl+C)
    print("\nProgram terminated by user.")

finally:
    # Clean up GPIO on program exit
    GPIO.cleanup()
