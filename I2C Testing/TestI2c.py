import serial
import time

# Configure the serial port
ser = serial.Serial(
    port='/dev/ttyAMA0',  # Use the primary UART port on Raspberry Pi 4
    baudrate=115200,       # Set the baud rate
    timeout=1            # Set a timeout value (in seconds) for read operations
)

count = 0
try:
    while True:
        #the current output string 
        output_string = f"{count}\n"
        # Send the number 4 as a string
        ser.write(output_string.encode('utf-8'))
        count +=1

        # Wait for a moment before sending the next number
        time.sleep(.01)

except KeyboardInterrupt:
    # Handle keyboard interrupt (Ctrl+C)
    print("\nProgram terminated by user")

finally:
    # Close the serial port when done
    ser.close()
