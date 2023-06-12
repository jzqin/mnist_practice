import numpy as np

def softmax(vals):
    vals = np.exp(vals) 
    vals = vals / np.sum(vals, axis=1, keepdims=True)
    return vals
