import cv2
image = cv2.imread('/Users/ryanzotti/Downloads/IMG_0302.JPG')
#image = cv2.imread('/Users/ryanzotti/Documents/repos/OpenCvDigits/images/cellphone.png')

#print(image.shape[0])
#print(image.shape[1])

# import the necessary packages
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from pyimagesearch.hog import HOG
from pyimagesearch import dataset
import argparse


# load the dataset and initialize the data matrix
(digits, target) = dataset.load_digits("/Users/ryanzotti/Documents/repos/OpenCvDigits/data/digits.csv")
data = []

# initialize the HOG descriptor
hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
    cellsPerBlock = (1, 1), normalize = True)

image = digits[0]
image = dataset.deskew(image, 20)
image = dataset.center_extent(image, (20, 20))

# loop over the images
for image in digits:
    # deskew the image, center it
    image = dataset.deskew(image, 20)
    image = dataset.center_extent(image, (20, 20))

    # describe the image and update the data matrix
    hist = hog.describe(image)
    data.append(hist)

# train the model
model = LinearSVC(random_state = 42)
model.fit(data, target)

# dump the model to file
joblib.dump(model, args["model"])