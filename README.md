# set2model
Author's code for the paper "Set2Model networks: learning discriminatively to learn generative models"

To use the models in Caffe, please see the layers package:
    - MeanHistLayer - AVG baseline
    - GaussHistLossLayer - Set2Model-Gauss, Set2Model-GMM layers

To reproduce the experiments, please:
 1) obtain the image archive from the authors
 2) use the prepare.py script to fill the LMDB database
 3) use the corresponding caffe-format deep network specifications for training and testing, where substitute caffe-root for CCC, database directory for XXX, some directory to store the snapshots for TTT