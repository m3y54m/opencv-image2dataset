import cv2
import os
import numpy as np

# Source path: /src/
SRC_PATH = os.path.dirname(os.path.abspath(__file__))
# Base path: /
BASE_PATH = os.path.join(os.path.dirname(SRC_PATH))
# Image path: /img/input.png
IMAGE_PATH = os.path.join(os.path.join(BASE_PATH, "img"), "input.png")
# Dataset path: /data/dataset.npz
DATASET_PATH = os.path.join(os.path.join(BASE_PATH, "data"), "dataset.npz")

MAX_CONTOUR_AREA = 500

# Load image
inputImage = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
# Extending the image for 10 pixels in each direction with color (value) of 255 (white)
inputImage = cv2.copyMakeBorder(
    inputImage, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
)
# Convert output image to color image
outputImage = cv2.cvtColor(inputImage, cv2.COLOR_GRAY2BGR)
tempImage = outputImage.copy()
# Clone the image
image = inputImage.copy()
# Thresholding to get the binary image
_, image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
# Remove noise
kernel = np.ones((4, 4), np.uint8)
image = cv2.erode(image, kernel, iterations=2)
image = cv2.dilate(image, kernel, iterations=2)
# Find all contours in the image
contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

numberOfValidShapes = 0
digitsPositionList = []
digitsImageList = []

# Loop over all contours (images of persian digits)
for contour in contours:
    # https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    # Find the center of the contour using moments (You can also use cv2.boundingRect or cv2.minEnclosingCircle and so on)
    moment = cv2.moments(contour)

    if moment["m00"] != 0:
        # Calculate the centroid of the contour
        x = int(moment["m10"] / moment["m00"])
        y = int(moment["m01"] / moment["m00"])
        # Size and coordinates of the square around the digit
        squareSize = 34
        squareStart = (x - squareSize // 2, y - squareSize // 2)
        squareEnd = (
            x + squareSize // 2,
            y + squareSize // 2,
        )
        # Get the area of the contour
        contourArea = cv2.contourArea(contour)

        if contourArea < MAX_CONTOUR_AREA:

            numberOfValidShapes += 1
            # Draw a square around the contour
            cv2.rectangle(outputImage, squareStart, squareEnd, (0, 255, 0), 1)
            # Crop current digit from the image
            currentDigitImage = inputImage[
                squareStart[1] : squareEnd[1], squareStart[0] : squareEnd[0]
            ]
            # Add the position of the digit to the list
            digitsPositionList.append([x, y])
            # Add the image of the digit to the list
            digitsImageList.append(currentDigitImage)


# Sort the digits images by their positions
digitsImageList = [
    digitImage for _, digitImage in sorted(zip(digitsPositionList, digitsImageList))
]
# Sort the digits positions
digitsPositionList = sorted(digitsPositionList)

# Initialize the Persian digits dataset
datasetImages = np.zeros((numberOfValidShapes, squareSize, squareSize), dtype=np.uint8)
datasetLabels = np.zeros((numberOfValidShapes, 1), dtype=np.uint8)

# Loop over the digits images
for item in range(len(digitsPositionList)):
    digitValue = item // 5
    # Add the digit image and label to the dataset
    datasetImages[item] = digitsImageList[item]
    datasetLabels[item] = digitValue
    # Put text at the digit position
    cv2.putText(
        outputImage,
        str(digitValue),
        (digitsPositionList[item][0] + 5, digitsPositionList[item][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )

# Save the dataset in a file
np.savez_compressed(DATASET_PATH, images=datasetImages, labels=datasetLabels)

# You can load the dataset from the file using np.load method:
#
# data = np.load(DATASET_PATH)
# datasetImages = data['images']
# datasetLabels = data['labels']

print(f"Number of valid digits found: {numberOfValidShapes}")

# Show images in separate windows
cv2.imshow(f"inputImage", inputImage)
cv2.imshow(f"processedImage", image)
cv2.imshow(f"outputImage", outputImage)
# Show the last image of the dataset
cv2.imshow(f"{datasetLabels[49]}", datasetImages[49])

# Save the ouput image in a file
cv2.imwrite(f"{BASE_PATH}/img/output.png", outputImage)

cv2.waitKey(0)
