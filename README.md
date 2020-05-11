# Methodology and software for quantifying trajectories in clinical datasets

## Specificity of clinical datasets

Large clinical datasets are becoming increasingly available for systematic mining of associations between phenotypics data. 
We assume that the real-life clinical data are characterized by the following features:
1) They contain mixed data types (continuous, binary, ordinal, categorical variables)
2) They typically contain missing values with non-random pattern across the data matrix
3) They do not have a uniquely defined labeling (part of the clinical variables can be used to define clinical groups, 
but this can be done in several meaningfull ways)

## Clinical trajectories

We assume that any clinical dataset represents a landscape of possible patient phenotypes of various and multivariate 
degree of clinical gravity, which can be accompanied by the details of applied treatment. 
We also assume a possibility of existence of clinical trajectories, i.e. clinically relevant sequences of partly ordered phenotypes 
possibly representing consequtive states of a developing disease phenotype and leading to some final states (i.e., a 
lethal outcome). We also assume that clinical trajectories might be characterized by branching structure, representing
some important bifurcations in the development of a disease. 

Extracting cellular trajectories is a widely used methodology of genomics data analysis. 
Quantifying and visualizing clinical trajectories represents a more challenging data mining problem due to the data specificity.

## Dimensionality reduction and manifold learning in clinical datasets

Here we develop an semi-supervised methodology of clinical data analysis, based on constructing and exploring the properties
of principal trees (PT), which is a non-linear generalization of Principal Component Analysis (PCA). Principal trees are 
constructed using ElPiGraph method, which has been previously exploited in determining branching trajectories in various genoomics 
datasets (in particular, in single cell omics data). 

The methodology takes into account the specificity of clinical data by providing tools for the following steps of clinical data analysis:

1) Univariate and multi-variate quantification of nominal variables
2) Several methods for missing values imputation including built-in benchmarking of the imputation methods
3) Set of state-of-the-art methods for manifold learning
4) Extracting clinical trajectories using principal tree approach
5) Computational tools for exploring the clinical datasets using the principal tree approach

The methodology is implemented in Python with some functionality using R packages.

## Installation

## Case studies

We demonstrate application of the methodology to two clinical datasets, one of moderate size (1700 patients) and one of relatively large size (100000 patients).

### Complications of myocardial infarction

The oridinal database was collected in the Krasnoyarsk Interdistrict Clinical Hospital (Russia) in 1992-1995 years. The original database and its description cab be downloaded from [[https://leicester.figshare.com/articles/Myocardial_infarction_complications_Database/12045261/1]]. It contains information about 1700 patients and 110 features characterizing the clinical phenotypes and 12 features representing possible complications of the myocardial infarction disease. 

### Diabetes data set from UCI Machine Learning Repository

The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. The dataset can be downloaded from [[https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008]]. The data matrix contains 100000 clinical cases of diabetis characterized by 55 attributes.
