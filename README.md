# Fashion parsing models in TensorFlow
1. Tensorflow implementation of [Fully Convolutional Networks for Semantic Segmentation](http://arxiv.org/pdf/1605.06211v1.pdf) (FCNs).
2. TensorFlow implementation of [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

The implementation is largely based on the reference code provided by the authors of the paper [link](https://github.com/shelhamer/fcn.berkeleyvision.org). 
1. [Prerequisites](#prerequisites)
2. [Results](#results)
3. [Observations](#observations)

## Prerequisites
 - pydensecrf installation in windows with conda: `conda install -c conda-forge pydensecrf`. For linux, use pip or something.
 - To train model simply execute `python FCN.py` or `python UNet.py`
 - To visualize results for a random batch of images use flag `--mode=visualize`
 - `debug` flag can be set during training to add information regarding activations, gradients, variables etc.
 - This [IPython notebook](https://github.com/shekkizh/FCN.tensorflow/blob/master/logs/images/Image_Cmaped.ipynb) can be used to view results in color as below.

## Results
Pretty much used the same network design as in the reference model implementation of the paper in caffe. The weights for the new layers added were initialized with small values, and the learning was done using Adam Optimizer (Learning rate = 1e-4). 

## Observations
For the gradients,
  - If you closely watch the gradients you will notice the inital training is almost entirely on the new layers added - it is only after these layers are reasonably trained do we see the VGG layers get some gradient flow. This is understandable as changes the new layers affect the loss objective much more in the beginning.
  - The earlier layers of the FCN netowrk are initialized with VGG weights and so conceptually would require less tuning unless the train data is extremely varied - which in this case is not.
  - The first layer of convolutional model captures low level information and since this entrirely dataset dependent you notice the gradients adjusting the first layer weights to accustom the model to the dataset.
  - The other conv layers from VGG have very small gradients flowing as the concepts captured here are good enough for our end objective - Segmentation. 
  - This is the core reason **Transfer Learning** works so well. Just thought of pointing this out while here.
