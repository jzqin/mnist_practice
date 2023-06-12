import numpy as np

class Layer:
    def __init__(self, n_in, n_out):
        self.lr = 0.05                # learning rate
        self.num_nodes = n_in         # number of hidden nodes in layer
        self.W = np.random.normal(0, np.sqrt(2 / (n_in + n_out)), (n_in, n_out)) # weight matrix
        self.b = np.zeros(n_out)      # bias vector
        self.activations = None       # store the final activations internally
        self.layer_norm_mean = None   # store some helpful layer norm values 
        self.layer_norm_var = None
        self.pre_ln_activations = None
        
    def _layer_norm(self, activations, predict):
        if not predict:
            self.pre_ln_activations = activations
        mean = np.mean(activations, axis=1, keepdims=True)
        var = np.var(activations, axis=1, keepdims=True)
        activations = (activations - mean) / (np.sqrt(var + 1e-10))
        
        if not predict:
            self.layer_norm_mean = mean
            self.layer_norm_var = var
        return activations
    
    def _layer_norm_jacobian(self):
        activations = self.pre_ln_activations
        n_batch = activations.shape[0]
        n_feat = activations.shape[1]
        std_dev = np.sqrt(self.layer_norm_var)
        jacobian = (np.eye(n_feat) * n_feat)
        jacobian = jacobian[np.newaxis,:,:]
        jacobian = np.repeat(jacobian, repeats=n_batch, axis=0)

        std_dev = np.repeat(std_dev, repeats=n_feat, axis=1)
        std_dev = np.repeat(std_dev[:,:,np.newaxis], repeats=n_feat, axis=-1)

        mean = np.repeat(self.layer_norm_mean, repeats=n_feat, axis=1)
        
        jacobian = jacobian / (std_dev*n_feat)
        activations = activations - mean
        activations = np.einsum('bi,bj->bij', activations, activations)
        # equivalently: activations = np.matmul(activations[:,:,np.newaxis), activations[:,np.newaxis,:])
        activations = activations / (n_feat * std_dev**3)
        jacobian = jacobian - activations
        return jacobian
        
    def forward(self, activations, predict=False):
        # import pdb; pdb.set_trace()
        activations = activations
        activations = np.matmul(activations, self.W) + self.b
        activations = self._layer_norm(activations, predict)
        activations = np.maximum(0, activations)
        if not predict:
            self.activations = activations
        return activations
    
    def backward(self, dx_next, prev_activations):
        if (self.activations is None):
            raise RuntimeError('Need to call forward pass for network before calling backward pass')

        relu_mask = self.activations
        relu_mask[relu_mask > 0] = 1
        relu_mask[relu_mask <= 0] = 0

        # import pdb; pdb.set_trace()
        dx_next = dx_next * relu_mask
        ln_jac = self._layer_norm_jacobian()
        dx_next = np.einsum('bi,bij->bj', dx_next, ln_jac)
        
        dx_curr = np.matmul(self.W, dx_next.T).T
        
        dW = np.einsum('bi,bj->bij', prev_activations, dx_next)
        dW = np.mean(dW, axis=0)
        self.W = self.W - (self.lr * dW)
        db = dx_next
        db = np.mean(db, axis=0)
        self.b = self.b - (self.lr * db)
        
        return dx_curr
                
    def get_weights(self):
        return self.W, self.b
    
    def get_activations(self):
        return self.activations
