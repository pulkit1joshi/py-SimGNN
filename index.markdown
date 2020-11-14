---
layout: post
title:  "SimGNN: Similarity Computation via Graph Neural Networks"
date:   2020-10-21 20:54:48 +0530
categories: [Deep Learning]
permalink: idk
description: This post will summarize the paper SimGNN which aims for fast graph similarity computation.
---

This post will summarize the paper SimGNN which aims for fast graph similarity computation. Graphs are structures that are used to link different entities that we call nodes using relationships called edges. Graphs exist everywhere from bonds between the atoms to friends on Facebook, all these scenarios can be represented as a graph. One of the fundamental graph problems includes finding similarity between graphs. The similarity between graphs can be defined using these metrics :

1.  Graph Edit Distance
2.  Maximum Common Subgraph

However, currently available algorithms that are used to calculate these metrics have high complexities and it is not yet possible to compute exact GED using these for graphs having more than 16 nodes.

Some ways to compute these metrics are :

1.  Pruning verification Framework
2.  Approximating the GED in fast and heuristic ways

SimGNN follows another approach to tackle this problem i.e turning similarity computation problem into a learning problem.

Before getting into how SimGNN works, we must know the requirements to be satisfied by this model. It includes :

1.  **Representation Invariant**: Different representations of the same graph should give the same results.
2.  **Inductive:** Should be able to predict results for unseen graphs.
3.  **Learnable:** Must work on different similarity metrics like GED and MCS

**SimGNN Approach:** To achieve the above-stated requirements, SimGNN uses two strategies

1.  Design Learnable Embedding Function: This maps the graph into an embedding vector, which provides a global summary of a graph. Here, some nodes of importance are selected and used for embedding computation. (less time complexity)
2.  Pair-wise node comparison: The above embedding are too coarse, thus further compute the pairwise similarity scores between nodes from the two graphs, from which the histogram features are extracted and combined with the graph level information. (this is a time-consuming strategy)

**Some Definitions :**

**Network Embedding** is defined as a collective term for techniques for mapping graph nodes to vectors of real numbers in a multidimensional space. To be useful, a good embedding should preserve the structure of the graph. The vectors can then be used as input to various network and graph analysis tasks, such as link prediction

**Graph Edit Distance** is one of the metrics for finding graph similarity. It is equal to the minimum number of edit operations required to convert one graph to another. The edit operations are :

1.  Adding vertex or removing vertex or vertex substitution
2.  Adding edge of removing edge or edge substitution

**SimGNN: Approach**
====================

SimGNN follows two strategies, one is graph level embedding and is faster, while the other is node-level interaction for better results but have higher time complexity. These two approaches are:

1.  **Graph Level Embedding Interaction**
2.  **PairWise Node Comparison**
![imge](https://miro.medium.com/max/700/0*MMk2UHltPTNMz6wH)

Ref: SimGNN paper

### **Graph Level Embedding Interaction:** 
In this, we encode our graph into a graph-level embedding that preserves the structural and feature information of the graph. Then further these graph level embeddings are compared to find the similarity between two graphs. The steps involved in this are :

#### 1.  **Node Level Embedding**  
GCN is used for node embedding. GCN is the architecture of GNN where the neighbor interaction is such that it learns an aggregation function. Nodes are hot encoded depending upon there types. Nodes having no label are given a new separate but the same category. Finally, these node embedding thus gained are fed into the attention module.

<p align="center">
<img src="https://miro.medium.com/max/153/1*v3RXIqJM9ALCMwuB_rVdJA.jpeg"/>
</p>


#### **2\. Graph Level Embedding:**

*   Now we have a set of node embedding and we can aggregate, do the weighted sum, to get one embedding per graph. But which nodes should get more weight? For this, the authors have presented an Attention mechanism. How does this work?
*   Let **U** be node embeddings of size **N X D**, where the nth row is embedding of the nth node which is **u** of size **D X 1**. Now we find an unweighted average of these **u** and calculate graph context **c** which is a D sized vector as :

<p align="center">
<img src="https://miro.medium.com/max/193/1*Uux9wyWrJPrslRiVzgEXXg.jpeg"/>
</p>


C is given as above

*   here **W** is of dimension **D X D** and **u** is of dimension **D X 1**. Here, the learnable parameter is W, and thus finding specific weights for every 1 x 1 element of **u,** thus in the end it becomes equivalent to finding elements of **W** such that they provide appropriate weights to each element of **u**
*   Further, to find attention **a(n)**, for every node, **u(n)** inner product **c** is found and mapped from 0 to 1 using the sigmoid function.


<p align="center">
<img src="https://miro.medium.com/max/254/1*ayBOz8_71gV_uxPpU0cJWA.jpeg"/>
</p>

a(n) i.e attention is found as above where f = sigmoid

*   Finally, graph embedding of size 1 x D is found as :  
    **h =** _∑_ **_a(n)u(n)_**


<p align="center">
<img src="https://miro.medium.com/max/1194/1*vwY6gmIFjGzIExUP_yuLPQ.jpeg"/>
</p>



<p align="center">
<img src="https://miro.medium.com/max/542/1*x8fvvSE9YH5l1ToI3twrNQ.jpeg"/>
</p>


### **Node level and Graph level embedding:**

#### **3\. Graph-Graph Interaction: Neural Tensor Network**

*   Now as we have graph embeddings of different graphs, we wish to find the similarity between them. This is done using a neural tensor network.
*   This gives a **K x 1** vector which is further fed into a neural network to find the similarity score.

<p align="center">
<img alt="Image for post" class="t u v jq aj" src="https://miro.medium.com/max/1042/1*KK5ZKMaJ6ylnuJL7q_qA-Q.jpeg"/>
</p>

**Node level, Graph level, and Neural tensor network**

*   I will be posting another article for NTNs ( [Link](https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf) ) and GNNs ( almost complete, based on this [Video](https://www.youtube.com/watch?v=zCEYiCxrL_0) ) soon, however, this can be regarded as a better way to find the relation between embeddings of two graphs.

#### **4\. Graph Similarity Score Computation**

Now having a K sized vector, the output is then fed into a fully connected layer and compared to ground-truth similarity and optimized using mean squared error loss function.


<p align="center">
<img src="https://miro.medium.com/max/696/1*VVsRO0HhK8Gk_twNu-YhoQ.jpeg"/>
</p>


**Loss function to be optimised:** Here s(i,j) = score between graph i and j

#### **PairWise Node Comparison**

This is a more heavy computational strategy, as compared to the previous one, where these steps are followed:

1.  **Computing the Graph similarity matrix**  
    Here every node embedding we gathered in step 1 is taken and multiplied with every other embedding ( including itself ) and then sigmoid activation is applied to get a N x N matrix where N = max(N1, N2) and N1, N2 are nodes of both the graphs. In the case of graphs with different nodes, extra nodes are added with zero columns.


<p align="center">
<img src="https://miro.medium.com/max/280/1*sxt7RD1MnZlvBiMtjkz8iw.jpeg"/>
</p>

**S**: Similarity matrix

**2\. Extracting histogram Features**

**3\. Concatenate and Feed**: Concatenating with the output of strategy 1 Neural Tensor Network obtained in the previous step and passing through a Fully connected layer.

Combining the two strategies, graph level embeddings and node to node interaction, as discussed above gives use following SimGNN:


<p align="center">
<img src="https://miro.medium.com/max/700/0*MMk2UHltPTNMz6wH"/>
</p>


### **Conclusion**

When different datasets are used with SimGNN and compared with ground truth values obtained from GED computation or the heuristic ways of GED gives some promising results. Along with the fast computation of similarity, SimGNN performs best among all the other ways on two datasets and second-best in the third data set. Further details can be read using the paper referred to below.

Keras Implementation can be found here: [https://github.com/pulkit1joshi/SimGNN](https://github.com/pulkit1joshi/SimGNN)/

***References:***  

*SimGNN: A Neural Network Approach to Fast Graph Similarity Computation. Yunsheng Bai, Hao Ding, Song Bian, Ting Chen, Yizhou Sun, Wei Wang. WSDM, 2019. [\[Paper\]](http://web.cs.ucla.edu/~yzsun/papers/2019_WSDM_SimGNN.pdf)*

*[https://github.com/pulkit1joshi/SimGNN](https://github.com/pulkit1joshi/SimGNN)*
