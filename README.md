# Methodology and software for quantifying trajectories in clinical datasets

Large clinical datasets are becoming increasingly available for systematic mining of associations between phenotypics data. 
We assume that the real-life clinical data are characterized by the following features:
1) They contain mixed data types (continuous, binary, ordinal, categorical variables)
2) They typically contain missing values with non-random pattern across the data matrix
3) They do not have a uniquely defined labeling (part of the clinical variables can be used to define clinical groups, 
but this can be done in several meaningfull ways)

We assume that any clinical dataset represents a landscape of possible patient phenotypes of various and multivariate 
degree of clinical gravity, which can be accompanied by the details of applied treatment. 
We also assume a possibility of existence of clinical trajectories, i.e. clinically relevant sequences of partly ordered phenotypes 
possibly representing consequtive states of a developing disease phenotype and leading to some final states (i.e., a 
lethal outcome). We also assume that clincal trajectories might be characterized by branching structure, representing
some important bifurcations in the development of a disease.

Here we develop an semi-supervised methodology of clinical data analysis, based on constructing and exploring the properties
of principal trees (PT), which is a non-linear generalization of Principal Component Analysis (PCA). Principal trees are 
constructed using ElPiGraph method, which has been previously exploited in determining branching trajectories in various genoomics 
datasets (in particular, in single cell omics data).
