# Importing the libraries
import logging
import os
from tabnanny import verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # 0 means no error, 1 means error, 2 means warning, 3 means info, 4 means debug. So 3 means we only want to see info, warning, error and critical messages, 2 means we only want to see warning, error and critical messages, etc.
import numpy as np
import tensorflow as tf
import sys


# parse command line

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # this line is required to call the main() function

# hide logs from tensorflow and keras: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)

# Pre processing
# __________________________________________________


def removeRedundantFeatures(dataset_input, memory_list):
    # Declare relevant Dataset
    relevant_data = []
    # Iterate through dataset
    for i in range(len(dataset_input)):
        relevant_data_point = []
        for j in range(len(dataset_input[i])):
            # If the memory list for datapoints at this index is 0, append it to the relevant datapoint list
            if memory_list[j] != 0:
                relevant_data_point.append(dataset_input[i][j])

        relevant_data.append(relevant_data_point)

    return relevant_data

def main():
    # set_tf_loglevel(logging.FATAL)
    memory_list = []
    with open("memorylist.txt", "r") as f:
        for line in f:
            memory_list.append(float(line.strip()))

    CLASSES = []
    ANSWER = 0
    # ___________________________________ preprocessing


    # Loads our saved model from folder
    model = tf.keras.models.load_model('saved_model.h5')
    
    # sys.stdin = open('input.txt', 'r') # uncomment for testing
    sys.stdin = open(0) # This line opens the standard input stream as a file #  uncomment for production
    file_input = sys.stdin.read() # This line reads the standard input stream as a string

    
    # test_data = np.fromstring(input_line, sep=' ').reshape(-1, 2352) # test data is numpy array of 2352 columns and as many rows as needed to fit the data
    test_data = np.fromstring(file_input, sep=' ').reshape(1, 2352) # test data is numpy array of 2352 columns and as many rows as needed to fit the data
    sys.stdin.close() # close the file

    # Preprocessing function
    # Remove redundant rows from test_data, reduce size from 2352 to 1917
    test_data_clean = np.array(removeRedundantFeatures(test_data, memory_list))
    

    test_data_clean= test_data_clean.reshape(1,1917) # Reshapes each data point to be 1 row and 1917 columns (supposed to be 0 and 1917 because it's meant to be a 1D array)
        
    ANSWER = np.argmax(model.predict(test_data_clean, verbose=0), axis=-1) # Predicts the class of the test data, and returns the index of the highest probability class)))
    # print(ANSWER) # Prints the class of the test
    sys.stdout.write(str(ANSWER[0])) # Writes the class of the test to stdout

if __name__ == '__main__':
    main()