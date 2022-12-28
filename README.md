# DBDT

Here we have collected the code to accompany the preprint submitted to Decision Support Systems "Efficient Fraud Detection using Deep Boosting Decision Tree"  

Folders:

- "Data": This folder contains some pny-size data for reproduce the fraud detection data experiments.

- "Script": This folder contains the code for deep boosting decision tree:

  * First, run the "DBDT-SGD" to run the baseline of deep boosting decision tree for balanced data. 

  * To run the "DBDT-AUC" to train the deep boosting decision tree only by AUC maximization, but it can not get fine result. 
  * To run the "DBDT-Com" to compositional train the deep boosting decision tree by AUC maximization and exponential minimization for fraud detection.

  * If you run all the experiments on multilayers with all the choices of training deep boosting decision tree as in our paper, "DBDT-Com-multilayers" will reproduce the results in the paper.
