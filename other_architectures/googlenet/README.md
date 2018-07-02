**PoseNet implementation in Keras for Visual Servoing**

This is a slight variant implementation of PoseNet. This model has been implemented using Keras with a tensorflow backend in mind.

The original paper by Alex Kendall, Matthew Grimes, Roberto Cipolla can be viewed [here](https://arxiv.org/abs/1505.07427)

The weights are of Imagenet instead of Places dataset originally used. 

The model can be run by calling `python 3output-googlenet.py`.

Please change the training dataset files by editing the lines with the file locations (lines 411-431) inside the file.


![Model with 3 outputs](https://raw.githubusercontent.com/arunavkonwar/DL-Gantry/master/other_architectures/googlenet/model-3output.png)
