# ClinTrajan tutorial 

![Trajan emperor, optimus princeps](https://github.com/auranic/ClinTrajan/blob/master/images/trajan.png)

Application of ClinTrajan to a clinical dataset consists in two parts:

1. [Quantification of the data]()
2. [Application of ElPiGraph]() and  [downstream analyses]()

Here we illustrate only the most basic analysis steps using the [dataset of myocardial infarction complications]().  In order to follow the tutorial, one has to download the [ClinTrajan git](https://github.com/auranic/ClinTrajan) and unpack locally. The easiest way is to run the tutorial is to run the code through the [short ClinTrajan tutorial Jupyter notebook](ClinTrajan_tutorial_short.ipynb). Alternatively, one can copy-paste and run the commands in any convinient Python environment. 

There exists also [complete Jupyter notebooks](), allowing one to reproduce all the analysis steps reported in the [ClinTrajan manuscript](https://arxiv.org/abs/2007.03788).


## Quantification of the data

The quantification functions of ClinTrajan are stored in the module **clintraj_qi**, so first of all we will import it. In addition, we import other modules of ClinTrajan and some other standard functions.

```
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from clintraj_qi import *
from clintraj_eltree import *
from clintraj_util import *
from clintraj_ml import *
from clintraj_optiscale import *
```

Now, let us import a dataset. It is assumed that the table is tab-delimited, contains only numbers or missing values in any row or column except for the first column (containing observation name) and the first row (containing variable names). Here it is assumed that certain column and rows of the table, containing too many missing values, have been eliminated. If the original dataset contains categorical nominal variables, they must be encoded first, using, for example, dummy encoding:

```
df = pd.read_csv('data/infarction/all_dummies.txt',delimiter='\t')
display(df)
quantify_nans(df)
```

As one can see, only 34% of rows have complete values for all columns.

We need to know what types different variables have, which can be 'BINARY', 'ORDINAL', 'CONTINUOUS:
```
variable_types, binary, continuous, ordinal = detect_variable_type(df,10,verbose=False)
```
We need to impute missing values in the table, but for this we need some quantification already. Simple univariate quantification can be done via:
```
dfq,replacement_info = quantify_dataframe_univariate(df,variable_types)
with open('temp.txt','w') as fid:
    fid.write(replacement_info)
```
Note that we has written down just in case the quantification parameters such that we could use them after imputation of missing values and restoring the data table to the initial variable scales.

Now we can impute the missing values. One of the simple idea is to compute SVD on the complete part of the data matrix and then project the data points with missing variables onto the principal components. The imputed value will be the value of the variable in the projection point.
```
dfq_imputed = SVDcomplete_imputation_method(dfq, variable_types, verbose=True,num_components=-1)
dequant_info = invert_quant_info(load_quantification_info('temp.txt'))
df_imputed = dequantify_table(dfq_imputed,dequant_info)
display(df_imputed)
```

Now, we are ready to quantify the data table. We will do it by applying optimal scaling to the ordinal values. 

```
df = remove_constant_columns_from_dataframe(df_imputed)
variable_names = [str(s) for s in df.columns[1:]]
X = df[df.columns[1:]].to_numpy()
X_original = X
X_before_scaling = X.copy()
X,cik = optimal_scaling(X,variable_types,verbose=True,vmax=0.6)
```
The output looks like this:

![](https://github.com/auranic/ClinTrajan/blob/master/images/optimal_scaling.png)

which means that the sum of squared correlations (Q2 value) between all quantified ordinal variables and between ordinal and numerical variables increased, and in the final correlation table one can see more significant correlations.

Congrats, now we have the data matrix *X* to which we can apply the [ElPiGraph](https://sysbio-curie.github.io/elpigraph/) algorithm! Note that we also kept the original matrix *X_original* and the list with the list of variables *variable_names* and with variable types *variable_types*.

## Application of [ElPiGraph](https://sysbio-curie.github.io/elpigraph/) to quantify branching pseudotime

Before applying ElPiGraph, let us first reduce the dimensionality of the dataset (more than 100!). [Some tests](https://github.com/j-bac/scikit-dimension) showed that the intrinsic dimensionality is much lower (around 12!), so let us apply Principal Component Analysis in order to reduce the dimension to this number. Note that we center all variables to zero mean and scale them to unit variance. Setting *svd_solver* to 'full' helps getting reproducible results (yes, by default, PCA is not reproducible!)

```
reduced_dimension = 12
X = scipy.stats.zscore(X)
pca = PCA(n_components=X.shape[1],svd_solver='full')
Y = pca.fit_transform(X)
v = pca.components_.T
mean_val = np.mean(X,axis=0)
X = Y[:,0:reduced_dimension]
```
Now we construct the principal tree. You can specify only one parameter for the number of nodes in the tree, but here we explicitly show the value of the other parameters which are close to the default ones.

```
nnodes = 50
tree_elpi = elpigraph.computeElasticPrincipalTree(X,nnodes,drawPCAView=True,
                                                  alpha=0.01,Mu=0.1,Lambda=0.05,
                                                  FinalEnergy='Penalized')
tree_elpi = tree_elpi[0]
# some additional pruning of the graph
prune_the_tree(tree_elpi)
# extend the leafs to reach the extreme data points
tree_extended = ExtendLeaves_modified(X, tree_elpi, Mode = "QuantDists", ControlPar = .5, DoSA = False)
```
This produces a simple plot with a projection of the principal tree on PCA plane. In 12D, the topology of the tree is more complex than it seams from 2D PCA projection!

![](https://github.com/auranic/ClinTrajan/blob/master/images/principal_tree.png)

In particular, in the linear 2D projection, we can see more or less clearly only two branches of the principal tree.

Before moving any further, we will need two partitionings of the data points, by proximity to the node of the graph in the multi-dimensional data space, and by the tree 'segment' (meaning a sequence of nodes in the tree without any branching point).

```
# paritioning the data by tree branches
vec_labels_by_branches = partition_data_by_tree_branches(X,tree_extended)
# paritioning the data by proximity to nodes
partition, dists = elpigraph.src.core.PartitionData(X = X, NodePositions = tree_elpi['NodePositions'], 
                                                    SquaredX = np.sum(X**2,axis=1,keepdims=1),
                                                    MaxBlockSize = 100000000, TrimmingRadius = np.inf
                                                    )
partition_by_node = np.zeros(len(partition))
for i,p in enumerate(partition):
    partition_by_node[i] = p[0]
```

In order to visualize the intrinsic geometry of the principal tree, and use it to project the data points from R<sup>N</sup> to R<sup>2</sup>, we can apply a version of force-directed layout to the principal tree (remember that a tree is a [planar graph](https://en.wikipedia.org/wiki/Planar_graph)!)

Let us visualize the tree with data points colored by the proximity to the tree segments:

```
fig = plt.figure(figsize=(8, 8))
visualize_eltree_with_data(tree_extended,X,X_original,v,mean_val,'k',variable_names,
                          Color_by_partitioning = True, visualize_partition = vec_labels_by_branches)
plt.show()
```
![](https://github.com/auranic/ClinTrajan/blob/master/images/principal_tree_segments.png)

