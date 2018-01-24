import glob
import os
import sys

import cv2
import numpy as np
from keras.models import load_model
from scipy.misc import imread
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
STRIDE = 4

# Check that the number of command line arguments is correct
if len(sys.argv) != 2:
    print('Usage:', sys.argv[0], '<trained model path>')
    sys.exit()

# Load the trained model
full_model_path = os.path.expanduser(sys.argv[1])
model = load_model(full_model_path)

# Create parameters for the blob detector that will be used to find hot spots on the prediction map
parameters = cv2.SimpleBlobDetector_Params()
parameters.minThreshold = 20
parameters.maxThreshold = 100
# Create the blob detector with the above parameters
blob_detector = cv2.SimpleBlobDetector_create(parameters)

# Loop forever (or until a keyboard interrupt)
while True:
    # List all correctly named image files in the temp folder
    image_names = glob.glob(TEMP_PATH + 'sim*.png')
    # Order them by modification time
    image_names.sort(key=os.path.getmtime)
    # Load the last image in the list
    image = imread(image_names[-1])

    # Slice up the image into windows
    image_slices = view_as_windows(image, (WINDOW_SIZE, WINDOW_SIZE, 3), STRIDE)
    # Flatten the window array so that there is only one non-window dimension
    image_slices_flat = np.reshape(image_slices, (-1,) + image_slices.shape[-3:])
    # Run a prediction on all of the windows at once
    predictions = model.predict(image_slices_flat)
    # Reshape the predictions to have the same initial two dimensions as the original list of image slices
    predictions_row_arranged = np.reshape(predictions, image_slices.shape[:2])

    # Find blobs in the heat map, which will be located around the location of the stop signs
    blob_key_points = blob_detector.detect(predictions_row_arranged)
    # Convert the key points to tuple positions
    blob_positions = [key_point.pt for key_point in blob_key_points]
    # Convert the points to comma-separated values that each make up one line
    positions_comma_separated = [position[0] + ' ' + position[1] for position in blob_positions]
    # Open a temporary file in the temp folder to write to
    with open(SIGN_TEMP_FILE_PATH, 'w') as temp_file:
        # Print each of the comma-separated lines to the file
        for line in positions_comma_separated:
            print(line, file=temp_file)
    # Rename the temp file to the actual storage file
    os.rename(SIGN_TEMP_FILE_PATH, SIGN_POSITION_FILE_PATH)
