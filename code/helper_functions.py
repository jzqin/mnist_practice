import numpy as np
import os

def softmax(vals):
    vals = np.exp(vals) 
    vals = vals / np.sum(vals, axis=1, keepdims=True)
    return vals

def load_data(dir_path, data_file, label_file):
    data_file = os.path.join(dir_path, data_file)
    with open(data_file, 'rb') as f:
        header = f.read(16)
        data = np.frombuffer(f.read(), dtype='uint8')
    data = np.reshape(data, (-1, 28*28))

    label_file = os.path.join(dir_path, label_file)
    with open(label_file, 'rb') as f:
        header = f.read(8)
        labels = np.frombuffer(f.read(), dtype='uint8')

    return data, labels
