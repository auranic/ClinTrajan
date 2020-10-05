# ClinTrajan tutorial 

![Trajan emperor, optimus princeps](https://github.com/auranic/ClinTrajan/blob/master/images/trajan.png)

Application of ClinTrajan to a clinical dataset consists in two parts:

1. [Quantification of the data](##quantification-of-the-data)
2. [Application of ElPiGraph](##application-of-elpigraph-to-quantify-branching-pseudotime) and  [downstream analyses](###downstream-analyses-using-elpigraph)

Here we illustrate only the most basic analysis steps using the [dataset of myocardial infarction complications](https://leicester.figshare.com/articles/dataset/Myocardial_infarction_complications_Database/12045261/3).  In order to follow the tutorial, one has to download the [ClinTrajan git](https://github.com/auranic/ClinTrajan) and unpack locally. The easiest way is to run the tutorial is to run the code through [this ClinTrajan tutorial Jupyter notebook](../ClinTrajan_tutorial.ipynb). Alternatively, one can copy-paste and run the commands in any convinient Python environment. 

There exist also [complete Jupyter notebooks](https://github.com/auranic/ClinTrajan/), allowing one to reproduce all the analysis steps reported in the [ClinTrajan manuscript](https://arxiv.org/abs/2007.03788).


## Quantification of the data

The quantification functions of ClinTrajan are stored in the module **clintraj_qi**, so first of all we will import it. In addition, we import other modules of ClinTrajan and some other standard functions.

```
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from clintraj_qi import *
from clintraj_optiscale import *
from clintraj_eltree import *
from clintraj_util import *
from clintraj_ml import *
```

Now, let us import a dataset. It is assumed that the table is tab-delimited, contains only numbers or missing values in any row or column except for the first column (containing observation name) and the first row (containing variable names). Here it is assumed that certain column and rows of the table, containing too many missing values, have been eliminated. If the original dataset contains categorical nominal variables, they must be encoded first, using, for example, dummy encoding:

```
df = pd.read_csv('data/infarction/all_dummies.txt',delimiter='\t')
display(df)
quantify_nans(df)
```

![](https://github.com/auranic/ClinTrajan/blob/master/images/table_import.png)

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

Now we can impute the missing values. One of simple ideas is to compute SVD on the complete part of the data matrix and then project the data points with missing variables onto the principal components. The imputed value will be the value of the variable in the projection point.
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

Before applying ElPiGraph, let us first reduce the dimensionality of the dataset (more than 100!). [Some tests](https://github.com/j-bac/scikit-dimension) showed that the intrinsic dimensionality is much lower (around 12!), so let us apply Principal Component Analysis in order to reduce the dimension to this number. Note that we center all variables to zero mean and scale them to unit variance. Setting *svd_solver* to 'full' helps getting reproducible results (yes, by default, PCA is not reproducible in sckit-learn!)

```
reduced_dimension = 12
X = scipy.stats.zscore(X)
pca = PCA(n_components=X.shape[1],svd_solver='full')
Y = pca.fit_transform(X)
v = pca.components_.T
mean_val = np.mean(X,axis=0)
X = Y[:,0:reduced_dimension]
```
Now we construct the [principal tree](https://www.mdpi.com/1099-4300/22/3/296). You can specify only one parameter for the number of nodes in the tree, but here we explicitly show the values of the other parameters which are close to the default ones.

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
This produces a simple plot with a projection of the principal tree on PCA plane. In 12D, the topology of the tree is more complex then it seams from 2D PCA projection!

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

In order to visualize the intrinsic geometry of the principal tree, and use it to project the data points from R<sup>N</sup> to R<sup>2</sup>, we can apply a version of [force-directed layout](https://en.wikipedia.org/wiki/Force-directed_graph_drawing) to the principal tree (remember that a tree is a [planar graph](https://en.wikipedia.org/wiki/Planar_graph)!)

Let us visualize the tree with data points colored by the proximity to the tree segments:

```
fig = plt.figure(figsize=(8, 8))
visualize_eltree_with_data(tree_extended,X,X_original,v,mean_val,'k',variable_names,
                          Color_by_partitioning = True, visualize_partition = vec_labels_by_branches)
plt.show()
```
![](https://github.com/auranic/ClinTrajan/blob/master/images/principal_tree_segments.png)

### Downstream analyses using ElPiGraph

Now let us visualize something more interesting using the principal tree. We will visualize all lethal cases of myocardial infarction complications, and by the thickness of the tree edges, we will visualize the mortality trend along various clinical trajectories. Note that in our table the variable *LET_IS_0* means *'no lethal outcome'*!

```
fig = plt.figure(figsize=(8, 8))
non_lethal_feature = 'LET_IS_0'
visualize_eltree_with_data(tree_extended,X,X_original,v,mean_val,'k',variable_names,
                          Color_by_feature=non_lethal_feature, Feature_Edge_Width=non_lethal_feature,
                           Invert_Edge_Value=True,Min_Edge_Width=10,Max_Edge_Width=50,
                           Visualize_Edge_Width_AsNodeCoordinates=True,cmap='winter')
plt.show()
```
![](https://github.com/auranic/ClinTrajan/blob/master/images/principal_tree_lethality.png )

Ok, let us do some more insightfull visualizations. Not let us highlight all patients with age <65 years having bronchyal asthma in anamnesis:

```
fig = plt.figure(figsize=(8, 8))
inds = np.where((X_original[:,variable_names.index('AGE')]<=65)&(X_original[:,variable_names.index('zab_leg_03')]==1))[0]
colors = ['k' for i in range(len(X))]
for i in inds:
    colors[i] = 'r'
visualize_eltree_with_data(tree_extended,X,X_original,v,mean_val,colors,variable_names,
                          highlight_subset=inds,Big_Point_Size=100,cmap='hot')
plt.show()
```

![](https://github.com/auranic/ClinTrajan/blob/master/images/principal_tree_asthma.png)

Further we want to quantify the pseudotime, but for this we need to define the root node. Here it should be the node corresponding to the least complicated case of myocardial infarction. Let us make yet another visualization in order to find it. For this we will visualize, using pie-charts, the proportion of complications in each node of the tree. The pie-chart will show the fraction of uncomplicated cases by black, and the fraction of complications by red. The size of the chart corresponds to the number of data points it represents (for which this is the closest node of the graph). Note that the lethal outcome is considered as - serious! - complication.

```
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1,1,1)
complication_vars = ['FIBR_PREDS','PREDS_TAH','JELUD_TAH','FIBR_JELUD',
                     'A_V_BLOK','OTEK_LANC','RAZRIV','DRESSLER',
                     'ZSN','REC_IM','P_IM_STEN']
inds_compl = [variable_names.index(a) for a in complication_vars]
lethal = 1-X_original[:,variable_names.index('LET_IS_0')]
has_complication = np.sum(X_original[:,inds_compl],axis=1)>0
inds = np.where((has_complication==0)&(lethal==0))[0]
colors = ['r' for i in range(len(X))]
for i in inds:
    colors[i] = 'k'
visualize_eltree_with_data(tree_extended,X,X_original,v,mean_val,colors,variable_names,
                          highlight_subset=inds,Big_Point_Size=2,Normal_Point_Size=2,showNodeNumbers=True)
add_pie_charts(ax,tree_extended['NodePositions2D'],colors,['r','k'],partition,scale=30)
plt.show()
root_node = 8
print('Root node=',root_node)
```

![](https://github.com/auranic/ClinTrajan/blob/master/images/principal_tree_complications.png)

As a result, we have just identified the node number 8 as the potential root because the proportion of complications in it was the least. Now we know from where to start the clinical trajectories and quantify pseudotime:

```
all_trajectories,all_trajectories_edges = extract_trajectories(tree_extended,root_node)
print(len(all_trajectories),' trajectories found.')
ProjStruct = project_on_tree(X,tree_extended)
PseudoTimeTraj = quantify_pseudotime(all_trajectories,all_trajectories_edges,ProjStruct)
```

Nine clinical trajectories have been finally identified. As in clustering, it is up to us to give some meaning for them though. One of the ways to do it, is to test if a set of clinical variables can be associated to some trajectories via non-linear regression which - in the case of binary variables - can be the logistic regression:

```
vars = ['ritm_ecg_p_01','ritm_ecg_p_02','ritm_ecg_p_04']
for var in vars:
    List_of_Associations = regression_of_variable_with_trajectories(PseudoTimeTraj,var,variable_names,
                                                                    variable_types,X_original,R2_Threshold=0.5,
                                                                    producePlot=True,
                                                                    Continuous_Regression_Type='gpr',
                                                                    verbose=True)
```
Here some of the detected associations are visualized. The actual data values are shown as red points, the probability given by logistic regression is shown as red line, and a simple sliding window smoothing of the data is shown as blue line.

![](https://github.com/auranic/ClinTrajan/blob/master/images/variable_associations.png)

Note that the scale (number of nodes) along each trajectory can be different. In general, the pseudotime quantified along different trajectories can not be directly compared because it can correspond to completely different physical time scale!

The results of regression application for several variables can be shown simultaneously at the same plot:
```
pstt = PseudoTimeTraj[1]
colors = ['r','b','g']
for i,var in enumerate(vars):
    vals = draw_pseudotime_dependence(pstt,var,variable_names,variable_types,X_original,colors[i],
                                               linewidth=3,draw_datapoints=False)
plt.legend()
plt.show()
```

![](https://github.com/auranic/ClinTrajan/blob/master/images/variable_associations_several.png)

Finally, the pseudotime can be used to visualize pseudo-temporal profiles of any quantity that can be derived from the data. For example, we can compute the cumulative function of hazard in order to visualize the lethality risks along different clinical trajectories. We will do this, by using the survival analysis library 'lifelines'.

```
import lifelines
from lifelines import SplineFitter
from lifelines import NelsonAalenFitter
from lifelines import KaplanMeierFitter
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','tab:pink','tab:green']

event_data = np.zeros((len(df),2))
events = 1-np.array(df['LET_IS_0'])
label = 'Death'

for i,pstt in enumerate(PseudoTimeTraj):
    points = pstt['Points']
    times = pstt['Pseudotime']
    for i,p in enumerate(points):
        event_data[p,0] = times[i]
        event_data[p,1] = events[p]

plt.figure(figsize=(8,8))

for i,pstt in enumerate(PseudoTimeTraj):
    TrajName = 'Trajectory:'+str(pstt['Trajectory'][0])+'--'+str(pstt['Trajectory'][-1])
    points = pstt['Points']
    naf = NelsonAalenFitter()
    T = event_data[points,0]
    E = event_data[points,1]
    naf.fit(event_data[points,0], event_observed=event_data[points,1],label=TrajName)  
    naf.plot_hazard(bandwidth=3.0,fontsize=20,linewidth=10,color=colors[i])
```

![](https://github.com/auranic/ClinTrajan/blob/master/images/hazard_pseudotime.png)

We can see that six out of nine trajectories are associated with significantly increasing lethality risk.