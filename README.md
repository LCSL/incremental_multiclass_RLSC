# Incremental Multiclass RLSC
Incremental regularized least squares for multiclass classification (RLSC) with recoding, extension to new classes and fixed update complexity.
Code for reproducing the MNIST experiments of our paper:

>Camoriano, R., Pasquale, G., Ciliberto, C., Natale, L., Rosasco, L. and Metta, G., 2017, May. Incremental robot learning of new objects with fixed update time. In 2017 IEEE International Conference on Robotics and Automation (ICRA) (pp. 3207-3214). IEEE.

Final version: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7989364

Pre-print: https://arxiv.org/abs/1605.05045

Copyright (c) 2017
Istituto Italiano di Tecnologia, Genoa, Italy
R. Camoriano, G. Pasquale, C. Ciliberto, L. Natale, L. Rosasco, G. Metta
All rights reserved.



# Abstract
   We consider object recognition in the context of lifelong learning, where a robotic agent learns to discriminate between a growing number of object classes as it accumulates experience about the environment. We propose an incremental variant of the Regularized Least Squares for Classification (RLSC) algorithm, and exploit its structure to seamlessly add new classes to the learned model. The presented algorithm addresses the problem of having an unbalanced proportion of training examples per class, which occurs when new objects are presented to the system for the first time. 
   We evaluate our algorithm on both a machine learning benchmark dataset and two challenging object recognition tasks in a robotic setting. Empirical evidence shows that our approach achieves comparable or higher classification performance than its batch counterpart when classes are unbalanced, while being significantly faster.

# Instructions
Clone and run main.m

To modify the experimental setting, edit "dataConf_MNIST_inc.m" and/or the "Experimental setup" section of "main.m"

# Note
Tested on MATLAB R2014b
