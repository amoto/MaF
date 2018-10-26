## Multi-hop Assortativities For Networks Classification

In this work we introduce the concept of multi-hop assortativities which captures correlations between nodes situated at two extremes of a randomly selected path of a given lenght. We use this assorativities as fingerprints for characterize networks and then performing high accurate graph classification task.

https://arxiv.org/abs/1809.06253

This code was tested on Debian GNU/Linux 8.11 (jessie), python 2.7 and Matlab R2016.

## Usage
### Example
This code load precomputed features from features/ folder and train a Support Vector Machine Classifier:
'''
python training_svm.py -f mutag_features.csv
'''
Similarly for Random Forest:
'''
python training_rf.py -f mutag_features.csv
'''

### Computing MaF features for the given datasets


## Datasets
We also provide 12 well stablished graph benchmark datasets in chemoinformatics and social networks.

### Chemoinformatic

* MUTAG: Is a nitro compounds dataset divided in two classes according to their mutageic activity on bacterium Salmonella Typhimurium.
* PTC: Dataset contains compounds labeled according to carcinogenicity on rodents, divided in two groups. Vertices are labeled by 20 atom types
* NCI1: From the National Cancer Institute (NCI), are two balanced dataset of chemical compounds screened for activity against non-small cell lung cancer and ovarian cancer cell. They have 37 and 38 categorical node labels respectively
* NCI109: From the National Cancer Institute (NCI), are two balanced dataset of chemical compounds screened for activity against non-small cell lung cancer and ovarian cancer cell. They have 37 and 38 categorical node labels respectively
* ENZYMES: Is a dataset of protein tertiary structures consisting of 600 enzymes from the BRENDA enzyme database. The task is to assign each enzyme to one of the 6 EC top-level classes.
* PROTEINS: Is a two classes dataset in which nodes are secondary structure elements (SSEs). Nodes are connected if they are contiguous in the aminoacid sequence. 

### Social Nets

* COLLAB: Is a scientific-collaboration dataset, where ego-networks of researchers that have collaborated together are constructed. The task is then determine whether the ego-collaboration network belongs to any of three classes, namely, High Energy Physics, Condense Mater Physics and Astro Physics.
* IMDB-BINARY: Is an ego-networks of actors that have appeared together in any movie. Graphs are constructed from Action and Romance genres. The task is identify which genre an ego-network graph belongs to.
* IMDB-MULTI: Is the same that the previous one, but consider three classes: Comedy, Romance and Sci-Fi.
* REDDIT-BINARY: Binary classification task of 2000 graphs where each network represents of online discussion threads from Reddit. The task is to identify whether a discussion is Question/Answer or Debate-based thread.
* REDDIT-MULTI-5K: Multi-class classification task of 5000 networks of online discussion threads from Reddit. The task is to identify which subreddit thread a given discussion belongs to.
* REDDIT-MULTI-12K: Same that the previous one, but with 12000 graphs. 

### Usage

For each data set X, the Matlab command
  load X
loads into the memory a struct array containing graphs, and a column vector lx containing 
a class label for each graph.
X(i).am is the adjacency matrix of the i'th graph, 
X(i).al is the adjacency list of the i'th graph, 
X(i).nl.values is a column vector of node labels for the i'th graph,
X(i).el (not always available) contains edge labels for the i'th graph.

Example: 
typing "load mutag" in MATLAB
loads an structure called "mutag" with 188 element array of graph structures and a column of 188 numbers, 
each of which indicates the class that the corresponding graph belongs to.
It also loads a 188x1 vector called "lmutag" which contains the ground truth labels for each graph

