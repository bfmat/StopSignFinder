import os
import sys

from keras.models import load_model

# An inference script for the stop sign location system that reads images, processes them, and writes locations of found
# stop signs to a file to be read by the simulation
# Created by brendon-ai, January 2018


# Path to the temporary folder where screenshots are stored
TEMP_PATH = '/tmp/'

# Check that the number of command line arguments is correct
if len(sys.argv) != 2:
    print('Usage:', sys.argv[0], '<trained model path>')
    sys.exit()

# Load the trained model
full_model_path = os.path.expanduser(sys.argv[1])
model = load_model(full_model_path)
