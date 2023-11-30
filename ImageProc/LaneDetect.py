import numpy as np
from picamera2 import Picamera2
import time
import cv2
import sys

camera = Picamera2()
camera_config = camera.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
camera.configure(camera_config)
camera.start()

def picam_load():
    time.sleep(0.1)
    im = camera.capture_array()
    cv2.imwrite("test.jpg", im)
    left, right = lr_detector(image_processor(im))
    print(left, right)

def image_file(filename):
    image = cv2.imread(filename)
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
    low_t = 50
    # second threshold for the hysteresis procedure 
    high_t = 150
    # applying canny edge detection and save edges in a variable
    edges = cv2.Canny(blur, low_t, high_t)

    cv2.imwrite("edges.jpg", edges)
    
    # since we are getting too many edges from our image, we apply 
    # a mask polygon to only focus on the road
    # Will explain Region selection in detail in further steps
    region = edges #region_selection(edges)
    
    return region

def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
    Parameters:
        image: we pass here the output from canny where we have 
        identified edges in the frame
    """
    # create an array of the same size as of the input image 
    mask = np.zeros_like(image)   
    # if you pass an image with more then one channel
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    # our image only has one channel so it will go under "else"
    else:
          # color of the mask polygon (white)
        ignore_mask_color = 255
    # creating a polygon to focus only on the road in the picture
    # we have created this polygon in accordance to how the camera was placed
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # performing Bitwise AND on the input image and mask to get only the edges on the road
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def lr_detector(proc_img):
    shape = proc_img.shape
    middle = int(shape[1] / 2)
    countLeft = 0
    left = 0
    countRight = 0
    right = 0
    for i in range(800, 900):
        for j in range(0, middle):
            if proc_img[i][j] == 255:
                countLeft += 1
                left += j
        for j in range(middle, shape[1]):
            if proc_img[i][j] == 255:
                countRight += 1
                right += j
    avgLeft = left / countLeft
    avgRight = right / countRight
    return avgLeft, avgRight

if __name__ == '__main__':
    # image = image_file('./test.jpg')
    # proc_img = image_processor(image)
    # print(proc_img.shape)
    # cv2.imwrite('test_proc.jpg', proc_img)
    # left, right = lr_detector(proc_img)
    # print(left, right)

    picam_load()

