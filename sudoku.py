"""

numpy is a library for scientific computing.
tensorflow is a machine learning framework used here to load a pre-trained model.
imutils is a package that provides convenience functions for working with OpenCV.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils
import mycanny





classes = np.arange(0, 10)

model = load_model('model-OCR.h5')

# print(model.summary())
input_size = 48


def get_perspective(img, location, height = 900, width = 900):
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Apply Perspective Transform Algorith
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    cv2.imwrite('result.jpg', result)
    return result


def find_board(img):
    """Takes an image as input and finds a sudoku board inside of the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
   # edged = cv2.Canny(bfilter, 30, 180)
    edged = mycanny.canny_edge_detection(gray,1.2,(5, 5),40,180)
    
    
    
    cv2.imwrite('outputMyCanny.jpg', edged)
    keypoints = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours  = imutils.grab_contours(keypoints)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None
    
    # Finds rectangular contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(img, location)
    return result, location


# split the board into 81 individual images
def split_boxes(board):
    """Takes a sudoku board and split it into 81 cells. 
        each cell contains an element of that board either given or an empty cell."""
    rows = np.vsplit(board,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255.0
            # cv2.imshow("Splitted block", box)
            # cv2.waitKey(50)
            boxes.append(box)
    cv2.destroyAllWindows()
    return boxes



# Read image
img = cv2.imread('sudoku2.jpg')


# extract board from input image
board, location = find_board(img)


gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
cv2.imwrite('outputMyCannygray.jpg', gray)
# print(gray.shape)
rois = split_boxes(gray)

rois = np.array(rois).reshape(-1, input_size, input_size, 1)

# get prediction
prediction = model.predict(rois)
# print(prediction)

predicted_numbers = []
# get classes from prediction
for i in prediction: 
    index = (np.argmax(i)) # returns the index of the maximum number of the array
    predicted_number = classes[index]
    predicted_numbers.append(predicted_number)

# print(predicted_numbers)

# reshape the list 
board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)

print(board_num)

#cv2.imshow("Input image", img)
# cv2.imshow("Board", board)
cv2.waitKey(0)
cv2.destroyAllWindows()