# Importing the libraries
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
    # ___________________________________ preprocessing


    # Loads our saved model from folder
    model = tf.keras.models.load_model('saved_model.h5')
    test_batch_filename= input("Please input filename containing data to be classified, 1 row per label returned: ")
    # parse command line 
    sys.stdin = open(test_batch_filename, 'r') # This line replaces the standard input with the input.txt file
    

    # The line belowLoads in test data from stdin, but using a hardcoded filename input.txt
    # Then reads in the test data from stdin=>input.txt file, 
    # And reshapes the data to be 2352 columns wide and as many rows as needed to fit the data
    test_data = np.loadtxt(sys.stdin).reshape(-1, 2352) # test data is numpy array of 2352 columns and as many rows as needed to fit the data
    
    sys.stdin.close() # close the file

    # Preprocessing function
    # Line below creates test_data_clean as a numpy array of 1917 columns (number of features in the model) 
    # and as many rows as needed to fit the data in the input file's number of predictions (3 in this case) 
    
    test_data_clean = np.array(removeRedundantFeatures(test_data, memory_list)) 
    

    for data_point in test_data_clean:
        data_point = data_point.reshape(1, 1917) # Reshapes each data point to be 1 row and 1917 columns (supposed to be 0 and 1917 because it's meant to be a 1D array)
        # Make predictions, store in list
        CLASSES.append(np.argmax(model.predict(data_point)))


    # Write output to communicate with the marker as concatenated string
    answer = ''.join(map(str, CLASSES))
    sys.stdout = open('output.txt', 'w') # This line replaces the standard output with the output.txt file
    sys.stdout.write(str(answer))
    sys.stdout.close() # close the file

if __name__ == '__main__':
    main()