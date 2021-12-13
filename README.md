# Supertuxkart
## This repository contains work to optimize the hyperparameters of the size of training set of images (utils.py), the step size of training (utils.py), the number of layers in the deep learning network (planner.py), the kernel sizes (planner.py), and the size of input/output channels (planner.py). Controller.py is a simple python controller which has the aim point as input, but planner.py learns to predict the aim point as input. The input to planner.py is size 3 for each of the R,G,B channels in the colored images in the training set, and outputs 2 channels to classifiy each prediction because aim point has a true size of 2x1.
### Installation
> git clone https://github.com/chasem151/supertuxkart
> pip3 install pytorch tensorflow
### Execution
> cd homework
> python -m controller trackname -v 
(tracknames include zengarden, lighthouse, hacienda, snowtuxpeak, cornfield_crossing, scotland, cocoa_temple)
### Train and run planner.py
> cd homework
> 
> python -m utils zengarden lighthouse hacienda snowtuxpeak cornfield_crossing scotland 
> 
> python -m train
> 
> python -m planner trackname -v
