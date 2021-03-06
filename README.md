# ExploreHierarchicalAttentionNetworks
This repo is for self learning and exploring the HAN model. The HAN architecture is introduced in [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) by Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola1 and Eduard Hovy.

The experiment is set up based on [Irvinglove's model implementation](https://github.com/Irvinglove/HAN-text-classification/blob/master/HAN_model.py) to explore the effect of dataset size and input dimension to the performance of the HAN model. In our experiment, we used the Amazon Product Review (referenced below).

### List of Files
 * ReadDataAmazon.py - read Amazon Product Review data into csv file
 * ReadData.py - pre-process csv data into input data for the HAN model. Each review is transformed from text into a _max_sent x max_word_ matrix
 * HanModel.py - [Irvinglove's HAN model implementation](https://github.com/Irvinglove/HAN-text-classification/blob/master/HAN_model.py)
 * Train.py - train model using training and validation set, built based on [HAN-text-classification Tutorial by Irvinglove](https://blog.csdn.net/Irving_zhang/article/details/77868620)
 * Predict.py - test model using testing data

### Reference and Acknowledgment
* [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) by Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola1 and Eduard Hovy<br />
* [HAN-text-classification Tutorial by Irvinglove](https://blog.csdn.net/Irving_zhang/article/details/77868620)<br />
* [HAN-text-classification Implementation by Irvinglove](https://github.com/Irvinglove/HAN-text-classification) <br />
* Amazon Product Reviews: R. He, J. McAuley. Modeling the visual evolution of fashion trends with one-class collaborative filtering. WWW, 2016 J. McAuley, C. Targett, J. Shi, A. van den Hengel. Image-based recommendations on styles and substitutes. SIGIR, 2015
