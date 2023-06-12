import numpy as np
from .layer import Layer
from . import helper_functions as hf

class FFN:
    def __init__(self, n_hidden_layers, n_hidden_size, n_feat, n_out):
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_size = n_hidden_size
        self.n_feat = n_feat
        self.n_out = n_out
        self.layers = [Layer(n_feat, n_hidden_size)]
        self.layers += [Layer(n_hidden_size, n_hidden_size) for _ in range(n_hidden_layers)]
        self.layers += [Layer(n_hidden_size, n_out)]

    def predict(self, data, predict=True):
        act = data
        for l in self.layers:
            act = l.forward(act, predict=predict)
        out = hf.softmax(act)
        return out
    
    def forward_pass(self, data, labels):
        out = self.predict(data, predict=False) 
        
        one_hot_labels = np.eye(self.n_out)[labels]
        nll = -1*np.log(out[np.where(one_hot_labels==1)]) # negative log loss
        dl = out - one_hot_labels                         # derivative of NLL w.r.t. softmax inputs

        return nll, dl
    
    def backward_pass(self, data, dloss):
        dx_next = dloss
        for i, l in reversed(list(enumerate(self.layers))):
            if i == 0:
                prev_act = data
            else:
                prev_act = self.layers[i-1].get_activations()
            
            dx_next = l.backward(dx_next, prev_act)
            
        return
