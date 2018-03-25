#!/usr/bin/env python3

import glob
import os
import sys

import cv2
import numpy as np
from keras.models import load_model
from scipy.misc import imread, imsave
from skimage.util import view_as_windows

# An inference script for the stop sign location system that reads images, processes them, and writes locations of found
# stop signs to a file to be read by the simulation
# Created by brendon-ai, January 2018


# Path to the temporary folder where screenshots are stored, ending with a slash
TEMP_PATH = '/tmp/'
# Path to the file where stop sign positions are written in a CSV format
SIGN_POSITION_FILE_PATH = TEMP_PATH + 'sign_positions.csv'
# Path to the temporary file for briefly writing to
SIGN_TEMP_FILE_PATH = TEMP_PATH + 'sign_temp.csv'
# Side length of the square windows to cut the images into
WINDOW_SIZE = 16
# Stride to move the window by along both dimensions
STRIDE = 16

# Check that the number of command line arguments is correct
if len(sys.argv) != 2:
    print('Usage:', sys.argv[0], '<trained model path>')
    sys.exit()

# Load the trained model
full_model_path = os.path.expanduser(sys.argv[1])
model = load_model(full_model_path)

image = imread('/home/brendonm/Desktop/StopSign.png')

# Slice up the image into windows
image_slices = view_as_windows(image, (WINDOW_SIZE, WINDOW_SIZE, 3), STRIDE)
# Flatten the window array so that there is only one non-window dimension
image_slices_flat = np.reshape(image_slices, (-1,) + image_slices.shape[-3:])
imsave('/home/brendonm/Desktop/spam.png', image_slices_flat[0])
# Run a prediction on all of the windows at once
predictions = model.predict(image_slices_flat)
# Reshape the predictions to have the same initial two dimensions as the original list of image slices
predictions_row_arranged = np.reshape(predictions, image_slices.shape[:2])
# Convert the floating-point array to unsigned 8-bit integers, multiplying it by 255 to scale it into the range of 0
# to 255, instead of 0 to 1
predictions_integer = np.array(predictions_row_arranged * 255, dtype=np.uint8)
imsave('/home/brendonm/Desktop/HeatMap.png', predictions_row_arranged)
