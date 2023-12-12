import serial
import numpy as np
from picamera2 import Picamera2
import time
import cv2
import sys

lane = 380
threshold = 10

print("Initializing Serial Port")
ser = serial.Serial(
   port='/dev/ttyAMA0',  # Use the primary UART port on Raspberry Pi 4
   baudrate=115200,       # Set the baud rate
   timeout=1            # Set a timeout value (in seconds) for read operations
)

print("Initializing Camera")
camera = Picamera2()
camera_config = camera.create_still_configuration(main={"size": (640, 480)}, display="main")
camera.configure(camera_config)
camera.start()
time.sleep(2)
print("Camera Ready")

def send_to_uart(left, right):
    """
    Send the left and right lane line coordinates over the UART Serial Connection.
    Sends it in the form of a single 4 byte integer, where the left lane line is the first 16 bits and the right lane line is the last 16 bits.
    Parameters:
        left: x coordinate of the left lane line
        right: x coordinate of the right lane line
    """
    left = int(left)
    right = int(right)
    print(left, right)

    num = (left << 16) | right
    print(num)
    msg = f'{num}\n'
    print(msg)
    print(f'Extracted:{1} {2}', num >> 16, num & 0x0000FFFF)
    ser.write(msg.encode('utf-8'))

def single_cam():
    """
    Test function to capture a single image
    """
    im = camera.capture_array()
    cv2.imwrite("test.jpg", im)
    left, right = lr_detector(image_processor(im))
    print(left, right)
    send_to_uart(left, right)

def cam_loop():
    """
    Continuously capture images from the camera and send the left and right lane line coordinates over the UART Serial Connection.
    """
    while True:
        try:
            im = camera.capture_array()
            left, right = lr_detector(image_processor(im))
            print(left, right)
            send_to_uart(left, right)
        except KeyboardInterrupt:
            print("Camera Input Stopped")
            send_to_uart(0, 1000) # Signal for Pololu to move to "Ready State"
            break

def image_file(filename):
    print('start')
    image = cv2.imread(filename)
    print('end')
    return image

def image_processor(image):
    """
    Process the input image to detect lane lines.
    Parameters:
        image: image of a road where one wants to detect lane lines
        (we will be passing frames of video to this function)
    """
    # convert the RGB image to Gray scale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # applying gaussian Blur which removes noise from the image 
    # and focuses on our region of interest
    # size of gaussian kernel
    kernel_size = 5
    # Applying gaussian blur to remove noise from the frames
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    
    # first threshold for the hysteresis procedure
    low_t = 100
    # second threshold for the hysteresis procedure 
    high_t = 250
    # applying canny edge detection and save edges in a variable
    # edges = cv2.Canny(blur, low_t, high_t)

    # Attempt Binary Edge Detection
    _, binary = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
    
    region = binary

    cv2.imwrite("edges.jpg", region)
    
    return region

def lr_detector(proc_img):
    shape = proc_img.shape
    middle = int(shape[1] / 2)
    countLeft = 0
    left = 0
    countRight = 0
    right = 0
    print('start')
    for i in range((lane - threshold), (lane + threshold)):
        for j in range(0, middle):
            if proc_img[i][j] == 255:
                countLeft += 1
                left += j
        for j in range(middle, shape[1]):
            if proc_img[i][j] == 255:
                countRight += 1
                right += j
    print('end')
    avgLeft = (left / countLeft if countLeft != 0 else 0)
    avgRight = (right / countRight if countRight != 0 else 640)
    return avgLeft, avgRight


if __name__ == '__main__':
    # image = image_file('./test.jpg')
    # proc_img = image_processor(image)
    # print(proc_img.shape)
    # cv2.imwrite('test_proc.jpg', proc_img)
    # left, right = lr_detector(proc_img)
    # print(left, right)

    # left, right = lr_detector(image_file('./edges.jpg'))
    # time.sleep(1)

    #single_cam()

    cam_loop()
    ser.close()

