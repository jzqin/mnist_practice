# mnist_practice
Self-implemented FFN for analysis of MNIST handwriting data

Author: Jason Qin

Last Updated: June 2023

I implement a FFN in NumPy for personal practice. The code for the FFN class is located in `code/ffn.py`. The FFN is constructed with `Layer` objects, which are defined in the `code/layers.py` file. The user can specify the number of layers in the FFN. 

Each layer consists of the following operations:
- Multiplication by weights: `y = Wx + b`, where x is an input matrix of dimensions `(batch_size, n_in)`, W is a weight matrix of dimension `(n_in, n_out)`, and b is a bias matrix of dimension `(n_out)`.
- ReLU activation
- LayerNorm: the activations are scaled within each layer according to the formula `x = (x - \mu) / \sqrt(\var)`

An example of creating and running a model is shown in `jq_ffn_model.ipynb`.

For comparison, I also train a PyTorch Transformer model to predict MNIST labels from the image data. This model is located in `pytorch_transformer_model.ipynb`.
