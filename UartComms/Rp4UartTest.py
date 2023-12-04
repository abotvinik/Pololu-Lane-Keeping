import serial
import time


# Configure the serial port
ser = serial.Serial(
   port='/dev/ttyAMA0',  # Use the primary UART port on Raspberry Pi 4
   baudrate=115200,       # Set the baud rate
   timeout=1            # Set a timeout value (in seconds) for read operations
)


try:
   while True:
       # Send the number 4 as a string
       ser.write(b'hello\n')


       # Wait for a moment before sending the next number
       time.sleep(1)


except KeyboardInterrupt:
   # Handle keyboard interrupt (Ctrl+C)
   print("\nProgram terminated by user")


finally:
   # Close the serial port when done
   ser.close()
