# MMDW
Max-margin DeepWalk

Code of IJCAI2016: "Max-Margin DeepWalk: Discriminative Learning of Network Representation"

## Datasets
  We evaluate MMDW with three datasets, including Cora, Citeseer and Wiki.
  * data/Edgelist/*_edgelist.txt: original edgelist file of network *.
  * data/Category/*_category.txt: the category list of vertices.
  * data/Net/*_net.txt: the matrix M of obtained from transition matrix A of a network.
  * data/vector/: the folder to save learnt vectors of vertices.
  * data/svm_model/:  the folder to save trained svm classifiers.
  * data/Bias/: the folder to save calculated biasVectors.
  * data/result/: the folder to classification results.

## Run
   Run the following command to learn max-margin DeepWalk: 
   
    java -jar mmdw.jar "dataset" "data_folder" "order_of_alphaBias" 
   
   Here is an example:
   
    java -jar mmdw.jar Cora data/ -3 
   

## Cite
  If you use the code, please cite this paper:
  
    Cunchao Tu, Weicheng Zhang, Zhiyuan Liu, Maosong Sun. Max-Margin DeepWalk: Discriminative Learning of Network Representation. International Joint Conference on Artificial Intelligence (IJCAI 2016).
