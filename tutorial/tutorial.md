# ClinTrajan tutorial 

![Trajan emperor, optimus princeps](https://github.com/auranic/ClinTrajan/blob/master/images/trajan.png)

Application of ClinTrajan to a clinical dataset consists in two parts:

1. [Quantification of the data]()
2. [Application of ElPiGraph]() and  [downstream analyses]()

Here we illustrate only the most basic analysis steps using the [dataset of myocardial infarction complications]().  In order to follow the tutorial, one has to download the [ClinTrajan git](https://github.com/auranic/ClinTrajan) and unpack locally. The easiest way is to run the tutorial is to run the code through the [short ClinTrajan tutorial Jupyter notebook](ClinTrajan_tutorial_short.ipynb). Alternatively, one can copy-paste and run the commands in any convinient Python environment. 

There exists also [complete Jupyter notebooks](), allowing one to reproduce all the analysis steps reported in the [ClinTrajan manuscript](https://arxiv.org/abs/2007.03788).


## Quantification of the data

The quantification functions of ClinTrajan are stored in the module **clintraj_qi**, so first of all we will import it:

```
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from clintraj_qi import *
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
```

Now we can impute the missing values. One of the simple idea is to compute SVD on the complete part of the data matrix and then project the data points with missing variables onto the principal components. The imputed value will be the value of the variable in the projection point.
```
dfq_imputed = SVDcomplete_imputation_method(dfq, variable_types, verbose=True,num_components=-1)
display(dfq_imputed)
```

This function produces the following output:

![](https://github.com/auranic/ClinTrajan/blob/master/images/imputation_svd.png)

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

Congrats, now we have the data matrix *X* to which we can apply the [ElPiGraph](https://sysbio-curie.github.io/elpigraph/) algorithm!

## Application of [ElPiGraph](https://sysbio-curie.github.io/elpigraph/) to quantify branching pseudotime