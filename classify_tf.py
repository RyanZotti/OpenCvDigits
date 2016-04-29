# USAGE
# python classify.py --model models/svm.cpickle --image images/umbc_zipcode.png

# import the necessary packages
from __future__ import print_function
from sklearn.externals import joblib
from pyimagesearch.hog import HOG
from pyimagesearch import dataset
import argparse
import mahotas
import cv2
from pyimagesearch import imutils
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True,
    help = "path to where the model will be stored")
ap.add_argument("-i", "--image", required = True,
    help = "path to the image file")
args = vars(ap.parse_args())

import tensorflow as tf
sess = tf.InteractiveSession()

x_ = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x_, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

saver = tf.train.Saver()
saver.restore(sess, "/Users/ryanzotti/Documents/repos/OpenCvDigits/tf_model/model.ckpt")

# load the model
model = joblib.load(args["model"])

# initialize the HOG descriptor
hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
    cellsPerBlock = (1, 1), normalize = True)

# load the image and convert it to grayscale
image = cv2.imread(args["image"])

scale_factor = 1
if image.shape[1]/600 > 0:
    scale_factor = int(image.shape[1]/600)  
newx,newy = int(image.shape[1]/scale_factor),int(image.shape[0]/scale_factor)
image = cv2.resize(image,(newx,newy))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur the image, find edges, and then find contours along
# the edged regions
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort the contours by their x-axis position, ensuring
# that we read the numbers from left to right
cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])

# loop over the contours
for (c, _) in cnts:

    # compute the bounding box for the rectangle
    (x, y, w, h) = cv2.boundingRect(c)

    # if the width is at least 7 pixels and the height
    # is at least 20 pixels, the contour is likely a digit
    if w >= 7 and h >= 20:
        # crop the ROI and then threshold the grayscale
        # ROI to reveal the digit
        roi = gray[y:y + h, x:x + w]
        thresh = roi.copy()
        T = mahotas.thresholding.otsu(roi)
        thresh[thresh > T] = 255
        thresh = cv2.bitwise_not(thresh)

        # deskew the image center its extent
        thresh = dataset.deskew(thresh, 20)
        thresh = dataset.center_extent(thresh, (20, 20))

        big_thresh = cv2.resize(thresh,(int(thresh.shape[1]*5),int(thresh.shape[0]*5)))
        print(str(thresh.shape[1])+" "+str(thresh.shape[0]))
        cv2.imshow("thresh2", big_thresh)
        cv2.imshow("thresh", thresh)

        thresh_tf = imutils.resize(thresh, height = 28, width = 28)
        my_image = np.array([thresh_tf])
        
        prediction=tf.argmax(y_conv,1)
        
        my_image = np.array(thresh_tf.reshape(1, 784))
        my_image = my_image / 255
        my_image = my_image.astype('Float32')
        
        digit = prediction.eval(feed_dict={x_: my_image,keep_prob: 1.0}, session=sess)[0]
        
        # extract features from the image and classify it
        print("I think that number is: {}".format(digit))

        # draw a rectangle around the digit, the show what the
        # digit was classified as
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(image, str(digit), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        #cv2.imshow("image", newimage)
        cv2.imshow("image", image)
        cv2.waitKey(0)