import cv2
import matplotlib.pyplot as plt
import numpy as np

def binary_process(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    blured_img = cv2.GaussianBlur(gray,(15,15),0)
    img = cv2.adaptiveThreshold(blured_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    kernel = np.ones((13,13),np.uint8)
    img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

    return img

def find_shapes(img,frame):
    contours, _ = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000 and area <  100000 :
            epsilon = 0.02 * cv2.arcLength(cnt, True)   # 2% of perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # Draw contour
            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 3)
            
            # Get shape name based on number of vertices
            vertices = len(approx)
            shape_name = ""
            if vertices == 3:
                shape_name = "Triangle"
            elif vertices == 4:
                shape_name = "Rectangle"
            elif vertices > 6:
                shape_name = "Circle"
            
            # Label the shape
            x,y = approx.ravel()[0], approx.ravel()[1] - 10
            cv2.putText(frame, shape_name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    return frame


webcam = cv2.VideoCapture(0)

while True:
    _,frame = webcam.read()

    binary_img = binary_process(frame)
    frame = find_shapes(binary_img,frame)

    cv2.imshow('webcam',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # Esc key for closing the window
        break


webcam.release()
cv2.destroyAllWindows()