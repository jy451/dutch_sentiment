import numpy as np
import re
from sklearn.preprocessing import LabelBinarizer
from tensorflow.contrib import learn
import pdb
import collections

#load data   
def load_data(data_file):
    X=[]
    y=[]
    with open(data_file) as f:
        lines=f.readlines()
        for line in lines:
            y.append(line.split("\t")[0].strip())
            X.append(line.split("\t")[1].strip())
    class_num=len(set(y))
    for i,label in enumerate(y):
        tmp=np.array([0]*class_num)
        tmp[int(label)]=1
        y[i]=tmp
    return [X,np.array(y)]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

