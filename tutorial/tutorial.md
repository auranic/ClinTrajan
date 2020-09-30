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



## Application of [ElPiGraph](https://sysbio-curie.github.io/elpigraph/)



