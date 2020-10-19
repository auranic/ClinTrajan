# ClinTrajan Python package
# 
# Copyright (C) 2020,  Curie Institute, 26 rue d'Ulm, 75005 Paris - FRANCE
# Copyright (C) 2020,  University of Leicester, University Rd, Leicester LE1 7RH, UK
# Copyright (C) 2020,  Lobachevsky University, 603000 Nizhny Novgorod, Russia
# 
# ClinTrajan is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# ClinTrajan is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
# See the GNU  Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public  
# License along with this library; if not, write to the Free Software  
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
# 
# ClinTrajan authors:
# Andrei Zinovyev: http://andreizinovyev.site
# Eugene Mirkes: https://github.com/mirkes
# Jonathan Bac: https://github.com/j-bac
# Alexander Chervov: https://github.com/chervov


import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2_contingency
import scipy.stats
from statsmodels.formula.api import ols
from numpy.lib.stride_tricks import as_strided
import math
import matplotlib.pyplot as plt

def associate_with_categorical_var(categorical_var,variable_name,variable,var_type,score_threshold=0.7,pvalue_threshold_parameter=1e-10,verbose=True,Negative_Enrichment=False,Minimal_number_of_points=5,produce_plot=False):
    # compute an association of a categorical variable categorical_var with another variableof type var_type
    # (chi-square : binary, ordinal and categorical and ANOVA for continuous)
    if var_type=='BINARY' or var_type=='CATEGORICAL':
        d = {'CATEGORICAL': categorical_var, 'VAR': variable} 
        df = pd.DataFrame(data=d)
        table = pd.crosstab(df['CATEGORICAL'], df['VAR'])
        var_vals = list(table.columns)
        categorical_vals = list(table.index)
        #print(categorical_vals)
        #print(var_vals)
        stat, p, dof, expected = chi2_contingency(table)
        if verbose:
            display(table)
            print(expected)
        devs = (table-expected)/(table+expected)
        devs = devs.to_numpy()
        table_array = table.to_numpy()
        list_of_associations = []
        for i in range(devs.shape[0]):
            for j in range(devs.shape[1]):
                if devs[i,j]>score_threshold and table_array[i,j]>=Minimal_number_of_points and not Negative_Enrichment:
                    if verbose:
                        print(str(categorical_vals[i])+'\t'+str(var_vals[j])+'\t{}'.format(devs[i,j]))
                    list_of_associations.append((categorical_vals[i],var_vals[j],devs[i,j]))
                if devs[i,j]<-score_threshold and expected[i,j]>=Minimal_number_of_points and Negative_Enrichment:
                    if verbose:
                        print(str(categorical_vals[i])+'\t'+str(var_vals[j])+'\t{}'.format(devs[i,j]))
                    list_of_associations.append((categorical_vals[i],var_vals[j],devs[i,j]))
        if produce_plot and len(list_of_associations)>0:
            unique_vals = np.unique(categorical_var)
            d = {'BRANCH NUMBER': unique_vals, 'observed': table.to_numpy()[:,1], 'expected': expected[:,1]} 
            df1 = pd.DataFrame(data=d)
            df1.plot(x="BRANCH NUMBER", y=["observed", "expected"], kind="bar")
            plt.title(variable_name,fontsize=20)
            plt.show()
    if var_type=='CONTINUOUS' or var_type=='ORDINAL':
        variable_normalized = scipy.stats.zscore(variable)
        d = {'CATEGORICAL': categorical_var, 'VAR': variable_normalized}        
        df = pd.DataFrame(data=d)
        categorical_vals = np.unique(categorical_var)
        #df = df.get_dummies(['CATEGORICAL'])
        #results = gls('VAR ~ C(CATEGORICAL)', data=df,hasconst=True).fit()
        results = ols('VAR ~ C(CATEGORICAL)', data=df).fit()
        if verbose:
            display(results.summary())
        aov_table = sm.stats.anova_lm(results, typ=2)
        if verbose:
            display(aov_table)
        pvals = np.asarray([v for v in results.pvalues])
        scores = np.asarray([v for v in results.params])
        if verbose:
            print(categorical_vals)
            print('P-values',pvals)
            print('Scores:',scores)
        list_of_associations = []
        for i,p in enumerate(pvals):
            if p<pvalue_threshold_parameter:
                if scores[i]>score_threshold and not Negative_Enrichment:
                    list_of_associations.append((categorical_vals[i],'CONTINUOUS',scores[i]))
                if scores[i]<-score_threshold and Negative_Enrichment:
                    list_of_associations.append((categorical_vals[i],'CONTINUOUS',scores[i]))
        if produce_plot and len(list_of_associations)>0:
            my_dict = {}
            unique_vals = np.unique(categorical_var)
            for val in unique_vals:
                my_dict[val] = variable[np.where(categorical_var==val)]
            fig, ax = plt.subplots()
            ax.boxplot(my_dict.values())
            ax.set_xticklabels(my_dict.keys())      
            plt.title(variable_name,fontsize=20)
            plt.show()

        stat = aov_table.F[0]
        p = aov_table['PR(>F)'][0]
    return list_of_associations,p,stat

def moving_weighted_average(x, y, step_size=.1, steps_per_bin=1,
                            weights=None):
    # This ensures that all samples are within a bin
    number_of_bins = int(np.ceil(np.ptp(x) / step_size))
    bins = np.linspace(np.min(x), np.min(x) + step_size*number_of_bins,
                       num=number_of_bins+1)
    bins -= (bins[-1] - np.max(x)) / 2
    bin_centers = bins[:-steps_per_bin] + step_size*steps_per_bin/2

    counts, _ = np.histogram(x, bins=bins)
    #print(bin_centers)
    #print(counts)
    vals, _ = np.histogram(x, bins=bins, weights=y)
    bin_avgs = vals / counts
    #print(bin_avgs)
    n = len(bin_avgs)
    windowed_bin_avgs = as_strided(bin_avgs,
                                   (n-steps_per_bin+1, steps_per_bin),
                                   bin_avgs.strides*2)
    
    weighted_average = np.average(windowed_bin_avgs, axis=1, weights=weights)
    return bin_centers, weighted_average

def firstNonNan(floats):
  for i,item in enumerate(floats):
    if math.isnan(item) == False:
      return i,item

def firstNanIndex(floats):
  for i,item in enumerate(floats):
    if math.isnan(item) == True:
      return i

def lastNonNan(floats):
  for i,item in enumerate(np.flip(floats)):
    if math.isnan(item) == False:
      return len(floats)-i-1,item

def fill_gaps_in_number_sequence(x):
    firstnonnan,val = firstNonNan(x)
    firstnan = firstNanIndex(x)
    if firstnan is not None:
        x[0:firstnonnan] = val
    lastnonnan,val = lastNonNan(x)
    if firstnan is not None:
        x[lastnonnan:-1] = val
        x[-1] = val
    #print('Processing',x)
    firstnan = firstNanIndex(x)
    while firstnan is not None:
        #print(x[firstNanIndex:])
        firstnonnan,val = firstNonNan(x[firstnan:])
        #print(val)
        firstnonnan = firstnonnan+firstnan
        #print('firstNanIndex',firstnan)
        #print('firstnonnan',firstnonnan)
        #print(np.linspace(x[firstnan-1],val,firstnonnan-firstnan+2))
        x[firstnan-1:firstnonnan+1] = np.linspace(x[firstnan-1],val,firstnonnan-firstnan+2)
        #print('Imputed',x)
        firstnan = firstNanIndex(x)
    return x
    
def get_matrix_of_association_scores(associations):
    #the argument is a dictionary of triples key(variable): (object_id,value_of_the_variable,score)
    keys = list(associations.keys())
    objects = []
    for key in keys:
        assoc_list = associations[key]
        for assoc in assoc_list:
            objects.append(assoc[0])
    objects = list(set(objects))
    matrix_of_scores = np.zeros((len(keys),len(objects)))
    for key in keys:
        assoc_list = associations[key]
        for assoc in assoc_list:
            obj = assoc[0]
            score = assoc[2]
            matrix_of_scores[keys.index(key),objects.index(obj)] = score
    return matrix_of_scores, keys, objects

def get_standard_color_seq():
    color_seq = [[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0],
             [1,0,0.5],[1,0.5,0],[0.5,0,1],
             [0.5,1,0],[0.5,0.5,1],[0.5,1,0.5],
             [1,0.5,0.5],[0,0.5,0.5],[0.5,0,0.5],
             [0.5,0.5,0],[0.5,0.5,0.5],[0,0,0.5],
             [0,0.5,0],[0.5,0,0],[0,0.25,0.5],
             [0,0.5,0.25],[0.25,0,0.5],[0.25,0.5,0],
             [0.5,0,0.25],[0.5,0.25,0],[0.25,0.25,0.5],
             [0.25,0.5,0.25],[0.5,0.25,0.25],[0.25,0.25,0.5],
             [0.25,0.5,0.25],[0.25,0.25,0.5],[0.25,0.5,0.25],
             [0.5,0,0.25],[0.5,0.25,0.25],
             [1,0,0.25],[1,0.25,0],[1,0.25,0.25],
             [0,1,0.25],[0.25,1,0],[0.25,1,0.25],
             [0,0.25,1],[0.25,0,1],[0.25,0.25,1],
             ]
    return color_seq

def remove_constant_columns_from_dataframe(df):
    # first column is assumed to be row id, not a value
    X = df[df.columns[1:]].to_numpy()
    inds = np.where(np.std(X,axis=0)>0)[0]
    print('Removing ',X.shape[1]-len(inds),'columns')
    inds=inds+1
    inds = np.array([0]+list(inds))
    df = df[df.columns[inds]]
    return df

def get_colorseq_for_column(df,column_name,color_seq=None):
    stcol = color_seq
    if color_seq is None:
        stcol = get_standard_color_seq()
    vals_unique_df = df[column_name].value_counts()
    vals_unique = vals_unique_df.index.to_list()
    vals_unique_freq = vals_unique_df.to_numpy()
    vals = df[column_name].to_list()
    new_vals = [stcol[vals_unique.index(val)] for val in vals]
    return new_vals, vals_unique,vals_unique_freq
        
def brokenstick_distribution(dim):
    distr = np.zeros(dim)
    for i in range(0,dim):
        distr[i]=0
        for j in range(i,dim):
            distr[i]=distr[i]+1/(j+1)
        distr[i]=distr[i]/dim
    return distr


