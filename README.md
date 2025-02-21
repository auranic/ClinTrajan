# Methodology and software for quantifying pseudotemporal trajectories in clinical datasets

## Clinical datasets

Large clinical datasets are becoming increasingly available for systematic mining the associations between phenotypic variables,
characterizing a particular disease. Specific properties of the clinical datasets represent challenges for applying the 
standard repertoire of machine learning methods. 

We assume that the real-life clinical data are characterized by the following features:
1) They contain mixed data types (continuous, binary, ordinal, categorical variables, censored data)
2) They typically contain missing values with non-uniform pattern across the data matrix
3) They do not have a uniquely defined labeling (part of the clinical variables can be used to define clinical groups, 
but this can be done in several meaningfull ways)

## Clinical trajectories, pseudotime, and dynamical phenotyping of synchronic data

We assume that any clinical dataset represents a landscape of possible patient phenotypes of various and multivariate 
degree of clinical gravity, which can be accompanied by the details of applied treatment. 
We also assume a possibility of existence of clinical trajectories, i.e. clinically relevant sequences of partly ordered phenotypes 
possibly representing consecutive states of a developing disease phenotype and leading to some final states (i.e., a 
lethal outcome). Each clinical trajectory can be characterized by its proper pseudotime which allows to quantitatively characterize the degree of progression along the trajectory. 
Each clinical variable can be plotted then as a function of pseudotime for a given clinical trajectory.
We also assume that clinical trajectories can be characterized by branching structure, representing
some important bifurcations in the development of a disease. 

Extracting cellular trajectories is a widely used methodology of data analysis in genomics, especially in studying certain highly dynamic phenomena such as differentiation or development. 

Quantifying and visualizing clinical trajectories represents a more challenging data mining problem due to the data specificity.

## Method of elastic principal graphs (ElPiGraph) for extracting and analyzing the clinical trajectories

Here we develop a semi-supervised methodology of clinical data analysis, based on constructing and exploring the properties
of principal trees (PT), which is a non-linear generalization of Principal Component Analysis (PCA). Principal trees are 
constructed using ElPiGraph method, which has been previously exploited in determining branching trajectories in various genoomics 
datasets (in particular, in single cell omics data). 

The methodology takes into account the specificity of clinical data by providing tools for the following steps of clinical data analysis:

1) Univariate and multi-variate quantification of nominal variables, including the original implementation of optimal scaling for ordinal variables
2) Several original methods for missing values imputation based on Singular Value Decomposition
3) Set of state-of-the-art methods for manifold learning
4) Partitioning the data accordingly to the branches of the principal tree (analogue of clustering) and associating the branches to clinical variables.
5) Extracting clinical trajectories using principal tree approach and associating the trajectories to clinical variables.
6) Visualization of clinical variables using principal trees
7) Pseudotime plots of clinical variables along clinical trajectories

The methodology is implemented in Python.

## Installation

For the moment the only way to use the package is to copy the .py files from the 'code' folder and make them available in the Python path

## Tutorial

A simplified tutorial showing the basic steps of ClinTrajan analysis is [available here](https://github.com/auranic/ClinTrajan/blob/master/tutorial/tutorial.md).
This tutorial is just better explained presentation of [this Jupyter notebook](ClinTrajan_tutorial.ipynb).

## Case studies

We demonstrate application of the methodology to two clinical datasets, one of moderate size (1700 patients) and one of relatively large size (100000 patients).

### Complications of myocardial infarction

The database was collected in the Krasnoyarsk Interdistrict Clinical Hospital (Russia) in 1992-1995 years. The original database and its description can be downloaded from https://leicester.figshare.com/articles/Myocardial_infarction_complications_Database/12045261/1. It contains information about 1700 patients and 110 features characterizing the clinical phenotypes and 12 features representing possible complications of the myocardial infarction disease. 

Two Jupyter notebooks provide the exact protocol of the analysis of this database.
In order to use them, download the content of the git and start the notebook from the git folder.

* [QI_Infarctus.ipynb](QI_Infarctus.ipynb) - notebook documenting quantification and imputation of the datatable, which consists of the steps
  1. Removing the columns containing more than 30% of missing values
  2. Removing the rows containing more than 20% of missing values
  3. Determining the complete part of the table
  4. Classifying variables into types (BINARY, ORDINAL, CONTINUOUS). The categorical variables are supposed to be converted using the standard dummy coding.
  5. Univariate variable quantification
  6. Using the quantified complete part of the table, compute SVD of an order corresponding to the intrinsic dimension estimate
  7. Project vectors with missing values into the space of obtained principal components, the imputed values are taken from the projection values.

* [PT_Infarctus.ipynb](PT_Infarctus.ipynb) - notebook documenting the analysis of the imputed table, using the methodology of principal trees. It contains of the following steps:
  1. Pre-processing the dataset by projecting it into the space of the first principal components. Conservative 'elbow rule'-based estimate for the number of top principal components is used in this case.
  2. [Defining the classes of patients](images/definition_of_classes_infarctus.png), by a separate analysis of dependent (complication and lethality cause) variables, using principal trees. Note: when viewed online, the notebook misses the image showing the introduced classification of the patients. This image can be found [here](images/definition_of_classes_infarctus.png).
  3. Constructing the principal tree, and post-processing it (pruning short edges and extending the external branches in order to avoid the border effects).
  4. Computing and visualizing associations of the principal tree branches with the patient classes.
  5. Determining the 'root node' of the tree, most associated to the 'no complication' class.
  6. Compute and visualize all trajectories from the root node to the leaf nodes.
  7. Compute and visualize associations of the trajectories with all variables
  8. Compute and visualize the connection of complication variables with trajectories.
  9. Using principal tree for visualization of various variables
  10. Applying a panel of 12 manifold learning methods to the dataset


### Diabetes readmission data set from UCI Machine Learning Repository

The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. The dataset can be downloaded from UCI repository at https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008 or from Kaggle at https://www.kaggle.com/brandao/diabetes. The data matrix contains 100000 hospitalization cases with patients suffering from diabetis characterized by 55 attributes.

Three Jupyter notebooks provide the exact protocol of the analysis of this database.
In order to use them, download the content of the git and start the notebook from the git folder.

* [Set_up_Diabetes_Data_for_Modeling.ipynb](Set_up_Diabetes_Data_for_Modeling.ipynb) - notebook providing the encoding of the diabetes readmission dataset
* [QI_Diabetes.ipynb](QI_Diabetes.ipynb) - notebook providing the details of quantification of the diabetes dataset
* [PT_Diabetes.ipynb](PT_Diabetes.ipynb) - notebook providing the details of the analysis of clinical trajectories for the diabetes dataset


## Use of ClinTrajAn in studying large-scale clinical datasets

ClinTrajAn was used to analyse two large-scale clinical datasets collected on the cohorts of COPD (Chronic Obstructive Pulmonary Disease) patients (3).
An elastic principal tree was fitted to the reduced space. 4,585 participants from COPDGene Phase 2 were used to test longitudinal trajectories. 2,652 participants from SPIROMICS tested external reproducibility. The study used cross-sectional data to create an elastic principal tree, where the concept of time is represented by distance on the tree. Six clinically distinct tree segments were identified that differed by lung function, symptoms, and CT features: 1) Subclinical (SC); 2) Parenchymal Abnormality (PA); 3) Chronic Bronchitis (CB); 4) Emphysema Male (EM); 5) Emphysema Female (EF); and 6) Severe Airways (SA) disease. Cross-sectional SPIROMICS data confirmed similar groupings. 5-year data from COPDGene mapped longitudinal changes onto the tree. 29% of patients changed segment during follow-up; longitudinal trajectories confirmed a net flow of patients along the tree, from SC towards Emphysema, although alternative trajectories were noted, through airway disease predominant phenotypes, CB and SA.

Another example of ClinTrajAn use is for [analysing multi-omics data for revealing the mechanisms of drug resistance](https://www.nature.com/articles/s41467-025-55914-x).


## References:

(1) [Albergante L, Mirkes E, Bac J, Chen H, Martin A, Faure L, Barillot E, Pinello L, Gorban A, Zinovyev A. Robust and scalable learning of complex intrinsic dataset geometry via ElPiGraph. 2020. Entropy 22(3):296](https://www.mdpi.com/1099-4300/22/3/296)

(2) [Golovenkin SE, Bac J, Chervov A, Mirkes EM, Orlova YV, Barillot E, Gorban AN, Zinovyev A. Trajectories, bifurcations and pseudotime in large clinical datasets: applications to myocardial infarction and diabetes data. GigaScience, Volume 9, Issue 11, November 2020, giaa128](https://doi.org/10.1093/gigascience/giaa128)

(3) [Bell AJ, Ram S, Labaki WW, Murray S, Kazerooni EA, Galban S, Martinez FJ, Hatt CR, Wang JM, Ivanov V, McGettigan P, Khokhlovich E, Maiorino E, Suryadevara R, Boueiz A, Castaldi P, Mirkes EM, Zinovyev A, Gorban AN, Galban CJ, Han MK. Temporal Exploration of COPD Phenotypes: Insights from the COPDGene and SPIROMICS Cohorts. Am J Respir Crit Care Med. 2024 ](https://doi.org/10.1164/rccm.202401-0127oc)
