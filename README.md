## Reconstructed DNN Model for Air Quality Prediction

-------------

Implement a DNN model for air quality prediction using reconstruction error proposed in [2].

1. dat：Data of the model. 
	- air_bj.mat: Air quality and meteorology data of one year, sampled once an hour, from 35 monitoring stations in Beijing. Location information(longitude and latitude) of stations.

2. lib: Library of the model. 
	- DeepLearnToolbox-BY RUBIO HUANG: A neural network toolbox.
	
3. src: Rec-DNN model.
	- knn_weight.m: Function of weighted k-nearest neighbors.
	- knn.m: Function of k-nearest neighbors.
	- rect_knn.m: DNN model to predict air quality, utilizing knn to reconstruct error. 
	- rect_knnw.m: DNN model to predict air quality, utilizing weighted knn to reconstruct error. 
	- rect_li: DNN model to predict air quality, utilizing linear regression to reconstruct error. 
	
Reference:

1. [Prediction as a candidate for learning deep hierarchical models of data](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6284) (Palm, 2012)
2. [Real-time Transportation Prediction Correction using Reconstruction Error in Deep Learning]
