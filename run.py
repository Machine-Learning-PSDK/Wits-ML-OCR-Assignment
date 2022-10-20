# Importing the libraries
from pydoc import doc
import numpy as np
import tensorflow as tf
import os
import sys


# parse command line

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    memory_list = []
    with open("memorylist.txt", "r") as f:
        for line in f:
            memory_list.append(float(line.strip()))

    CLASSES = []
    ANSWER = 0
    # ___________________________________ preprocessing


    # Loads our saved model from folder
    model = tf.keras.models.load_model('saved_model.h5')
    
    sys.stdin = open('input.txt', 'r') # uncomment for testing
    # sys.stdin = open(0) # This line opens the standard input stream as a file #  uncomment for production
    file_input = sys.stdin.read() # This line reads the standard input stream as a string

    
    # test_data = np.fromstring(input_line, sep=' ').reshape(-1, 2352) # test data is numpy array of 2352 columns and as many rows as needed to fit the data
    test_data = np.fromstring(file_input, sep=' ').reshape(1, 2352) # test data is numpy array of 2352 columns and as many rows as needed to fit the data
    sys.stdin.close() # close the file

    # Preprocessing function
    # Remove redundant rows from test_data, reduce size from 2352 to 1917
    test_data_clean = np.array(removeRedundantFeatures(test_data, memory_list))
    

    test_data_clean= test_data_clean.reshape(1,1917) # Reshapes each data point to be 1 row and 1917 columns (supposed to be 0 and 1917 because it's meant to be a 1D array)
        
    ANSWER = np.argmax(model.predict(test_data_clean), axis=-1) # Predicts the class of the test data, and returns the index of the highest probability class)))
    # print(ANSWER) # Prints the class of the test
    sys.stdout.write(str(ANSWER[0])) # Writes the class of the test to stdout

if __name__ == '__main__':
    main()