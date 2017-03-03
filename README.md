# incremental_multiclass_RLSC
Incremental regularized least squares for multiclass classification with recoding, extension to new classes and fixed update complexity.

Code for the experiments of the paper on the MNIST handwritten digits dataset:
   "Incremental Robot Learning of New Objects with Fixed Update Time"
   Raffaello Camoriano, Giulia Pasquale, Carlo Ciliberto, Lorenzo Natale, Lorenzo Rosasco, Giorgio Metta
   ICRA 2017
 
Abstract
   We consider object recognition in the context of lifelong learning, where a robotic agent learns to discriminate between a growing number of object classes as it accumulates experience about the environment. We propose an incremental variant of the Regularized Least Squares for Classification (RLSC) algorithm, and exploit its structure to seamlessly add new classes to the learned model. The presented algorithm addresses the problem of having an unbalanced proportion of training examples per class, which occurs when new objects are presented to the system for the first time. 
   We evaluate our algorithm on both a machine learning benchmark dataset and two challenging object recognition tasks in a robotic setting. Empirical evidence shows that our approach achieves comparable or higher classification performance than its batch counterpart when classes are unbalanced, while being significantly faster.

Copyright (c) 2017
Istituto Italiano di Tecnologia, Genoa, Italy
R. Camoriano, G. Pasquale, C. Ciliberto, L. Natale, L. Rosasco, G. Metta
All rights reserved.

Clone and run main.m

To modify the experimental setting, edit "dataConf_MNIST_inc.m" and/or the "Experimental setup" section of "main.m"

Tested on MATLAB R2014b