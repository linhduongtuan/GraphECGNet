# GraphECGNet

*In this respect, our paper has the following contributions:*

- A practical approach to detect edges of waveform in ECG images, using the Sobel operator.

- A workable solution to the classification of heart diseases, deploying GNN techniques on ECG signals. To the best of our knowledge, our study is the first attempt to deploy GNNs in automatically classifying ECG signals for detecting heart problems.

- An empirical evaluation on the two real ECG datasets to compare our proposed model with two state-of-the-art approaches using convolutional neural networks (ResNet26D and ConvNet).

```
# File Structure and Working procedure
```
1. Fist, convert ECG signals from PTB-XL database to images using signal2image.py
2. Then, apply edge detection accroding to the class-number: edge_transformation.py
3. Afterwards, prepare graph-datasets using edge-preparation: Graph_construction.py
4. Finally edge preperation produces five kinds of dataset for graph classification:
  path name: .../GraphTrain/dataset/<dataset_name>/raw/<dataset_name>_<data_file>.txt. 
  <data_file> can be:
    
    4.1. A--> adjancency matrix 
    4.2. graph_indicator--> graph-ids of all node 
    4.3. graph_labels--> labels for all graph 
    4.4. node_attributes--> attribute(s) for all node 
    5.5. node_labels--> labels for all node
5. After all the graph datasets are created properly, run main.py. The graph datasets are loaded through dataloader.py and the model is called through model.py
```

# Citation
We have published our work entitled as "Edge detection and graph neural networks to classify mammograms: A case study with a dataset from Vietnamese patients" under the "Applied Soft Computing Journal". If this repository helps you in your research in any way, please cite our paper:
```bibtex
@article{DUONG2023120107,
title = {Fusion of edge detection and graph neural networks to classifying electrocardiogram signals},
journal = {Expert Systems with Applications},
pages = {120107},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.120107},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423006097},
author = {Linh T. Duong and Thu T.H. Doan and Cong Q. Chu and Phuong T. Nguyen},
keywords = {Deep learning, Graph neural network(s), Electrocardiogram (ECG/EKG), Bio-signalling, Healthcare},
abstract = {The analysis of electrocardiogram (ECG) signals are among the key factors in the diagnosis of cardiovascular diseases (CVDs). However, automatic processing of ECG in clinical practice is still restrained by the accuracy of existing algorithms. Deep learning methods have recently achieved striking success in a variety of task including predictive healthcare. Graph neural networks are a class of machine learning algorithms which can learn by directly extracting important information from graph-structured data, and perform prediction on unknown data. Such algorithms are suitable for mining complex graph data, deducing useful predictions. In this work, we present a Graph Neural Network (GNN) model trained in two datasets with more than 107,000 single-lead signal images extracted from laboratories of Boston’s Beth Israel Hospital and of the Massachusetts Institute of Technology (MITBIH), and 1.5 million labelled exams analyzed by the Physikalisch-Technische Bundesanstalt (PTB). Our proposed GNN achieves promising performance, i.e., the results show that ECG classification based on GNNs using either single-lead or 12-lead setup is closer to the human-level in standard clinical practice. By several testing instances, the proposed approach obtains an accuracy of 1.0, thereby outperforming various state-of-the-art baselines by both databases with respect to effectiveness and timing efficiency. We anticipate that the approach can be deployed as a non-invasive pre-screening tool to assist doctors in real-time monitoring and performing their diagnosis activities.}
}
```
### Latest DOI

[![DOI](https://doi.org/10.1016/j.eswa.2023.120107)](https://doi.org/10.1016/j.eswa.2023.120107)
