import sys
import glob
import os
import numpy as np
import cv2
import easyocr
import re

# Configure dataset path
DATASET_PATH = '/Users/anshulnimmmala/Desktop/car data/'
SEARCH_PATH = "/Users/anshulnimmmala/Desktop/search image/"

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Function to simulate gate opening (visual feedback)
def open_gate():
    print("Gate is opening...")
    gate_open_img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(gate_open_img, "GATE OPEN", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
    cv2.imshow("Gate Status", gate_open_img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

# Function to simulate gate remaining closed (visual feedback)
def close_gate():
    print("Gate remains closed. Access denied.")
    gate_closed_img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(gate_closed_img, "GATE CLOSED", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Gate Status", gate_closed_img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

# Detecting number plate
def number_plate_detection(img):
    def clean2_plate(plate):
        gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
        num_contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if num_contours:
            contour_area = [cv2.contourArea(c) for c in num_contours]
            max_cntr_index = np.argmax(contour_area)
            max_cnt = num_contours[max_cntr_index]
            max_cntArea = contour_area[max_cntr_index]
            x, y, w, h = cv2.boundingRect(max_cnt)

            if not ratioCheck(max_cntArea, w, h):
                return plate, None

            final_img = thresh[y:y+h, x:x+w]
            return final_img, [x, y, w, h]
        else:
            return plate, None

    def ratioCheck(area, width, height):
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        return not (area < 1063.62 or area > 73862.5) and (3 <= ratio <= 6)

    def isMaxWhite(plate):
        return np.mean(plate) >= 115

    def ratio_and_rotation(rect):
        (x, y), (width, height), rect_angle = rect
        angle = -rect_angle if width > height else 90 + rect_angle
        return angle <= 15 and height > 0 and width > 0 and ratioCheck(height * width, width, height)

    img2 = cv2.GaussianBlur(img, (5, 5), 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=3)
    _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, element)
    num_contours, _ = cv2.findContours(morph_img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in num_contours:
        min_rect = cv2.minAreaRect(cnt)
        if ratio_and_rotation(min_rect):
            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = img[y:y+h, x:x+w]
            if isMaxWhite(plate_img):
                clean_plate, rect = clean2_plate(plate_img)
                if rect:
                    x1, y1, w1, h1 = rect
                    x, y, w, h = x + x1, y + y1, w1, h1

                    # ✅ FIXED: Use NumPy array directly
                    result = reader.readtext(clean_plate)
                    if result:
                        text = result[0][1]
                        return text.strip() if text else None
    return None

# Quick sort algorithm
def quickSort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quickSort(left) + middle + quickSort(right)

# Binary search algorithm
def binarySearch(arr, x):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            high = mid - 1
        else:
            low = mid + 1
    return -1

print("HELLO!!")
print("Welcome to the Number Plate Detection System.\n")

array = []

# Process dataset for vehicle numbers
for img_path in glob.glob(DATASET_PATH + "*.jpeg"):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (600, 600))
    cv2.imshow("Image of car", img_resized)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    number_plate = number_plate_detection(img)
    if number_plate:
        res2 = "".join(re.split("[^a-zA-Z0-9]*", number_plate)).upper()
        print("Detected Number Plate:", res2)
        array.append(res2)

# Sorting
array = quickSort(array)
print("\n\nThe Vehicle numbers registered are:")
for i in array:
    print(i)
print("\n\n")

# Searching for number plates in search images
search_images = glob.glob(SEARCH_PATH + "*.jpeg")
if not search_images:
    print("No search images found in the directory.")
    close_gate()
    res2 = None  

else:
    res2 = None
    for img_path in search_images:
        img = cv2.imread(img_path)
        number_plate = number_plate_detection(img)
        if number_plate:
            res2 = "".join(re.split("[^a-zA-Z0-9]*", number_plate)).upper()
            print("Detected Number Plate in Search Image:", res2)
            break  

# Check if the detected number plate is valid
if not res2:
    print("No valid number plate detected in the search images.")
    close_gate()
else:
    # Check if the detected number plate is in the database
    result = binarySearch(array, res2)
    if result != -1:
        print("\n\nThe Vehicle is allowed to visit.")
        open_gate()
    else:
        print("\n\nThe Vehicle is not allowed to visit.")
        close_gate()
