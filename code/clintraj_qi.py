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
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def quantify_ordinal_vector(ord_var, center=True):
    nd = norm()
    ord_var1 = ord_var[~np.isnan(ord_var)]
    unv, unv_counts = np.unique(ord_var1, return_counts=True)
    pis = unv_counts/np.sum(unv_counts)
    pis_cum = np.zeros(len(pis))
    for i,v in enumerate(pis[:-1]):
        pis_cum[i+1] = pis_cum[i]+pis[i]
    for i,v in enumerate(pis[:-1]):
        pis_cum[i] = pis_cum[i] + pis[i]/2
    unv_quantified = nd.ppf(pis_cum)
    replacement_map = {}
    for i,val in enumerate(unv):
        val = int(val)
        replacement_map[val] = unv_quantified[i]
    ord_var_quantified = ord_var
    for i,v in enumerate(unv):
        ord_var_quantified = np.where(ord_var_quantified==v, unv_quantified[i], ord_var_quantified)
    if center:
        mean_val = np.mean(ord_var_quantified[~np.isnan(ord_var_quantified)])
        ord_var_quantified = ord_var_quantified-mean_val
        for x in replacement_map:
            val = replacement_map[x]
            val = val-mean_val
            replacement_map[x] = val
    return ord_var_quantified, replacement_map

def quantify_dataframe_univariate(df,variable_types):
    dfq = df.copy()
    dframe = dfq
    global_replacement_info = ''
    for i,v in enumerate(variable_types):
    # for continuous values, we normalize to z-score
        if v=='CONTINUOUS':
            col = dframe.columns[i+1]
            vals = dframe[col].to_numpy()
            vals1 = vals[~np.isnan(vals)]
            if np.std(vals1)>1e-10:
                vals = (vals-np.mean(vals1))/np.std(vals1)
            else:
                vals = (vals-np.mean(vals1))
            for i,v in enumerate(vals):
                dframe.loc[i,col] = v
            global_replacement_info = global_replacement_info+col+'\tCONTINUOUS\t{\'mean\':'+str(np.mean(vals1))+',\'std\':'+str(np.std(vals1))+'}\n'
    # for ordinal and binary values, we normalize the variable 
    # to simple average probability values
        if v=='ORDINAL' or v=='BINARY':
            col = dframe.columns[i+1]
            vals = dframe[col].to_numpy()
            vals_quantified,repl_map = quantify_ordinal_vector(vals)
            dframe[col].replace(repl_map,inplace=True)
            global_replacement_info = global_replacement_info+col+'\t'+v+'\t'+str(repl_map)+'\n'
    return dfq, global_replacement_info

def dequantify_table(dfq,dequant_info,verbose=False):
    df = dfq.copy()
    eps = 0.0001
    for col in df.columns[1:]:
        dq_info = [di for di in dequant_info if di[0]==col]
        if len(dq_info)==0:
            raise Exception('for the column '+col+' dequantification is not found!!!')
        else:
            dq_info=dq_info[0]
        if verbose:
            print(dq_info)
        if dq_info[1]=='CONTINUOUS':
            mean_value = dq_info[2]['mean']
            std_value = dq_info[2]['std']
            for i,v in enumerate(df[col]):
                df.loc[i,col] = v*std_value+mean_value
        else:
            repl_dict = dq_info[2]
            #print(repl_dict)
            for i,v in enumerate(df[col]):
                for kv in repl_dict:
                    if abs((kv-v)/(kv+v))<eps:
                        df.loc[i,col] = int(repl_dict[kv])
                        
            df = df.astype({col: 'int32'})
    return df


def binary_naive_imputation(datafr,redundant_pair):
    # To be implemented
    return datafr

def load_quantification_info(file_name):
    # returns list of tuples (var_name,var_type,map), where map is the replacement 
    # map for discrete variables and the map {'mean': mean_value, 'std': standard_deviation} 
    # for continuous numerical variables
    lines = []
    quant_info = []
    with open(file_name) as fid:
        lines = fid.readlines()
    for line in lines:
        line = line.strip('\n')
        parts = line.split('\t')
        var_name = parts[0]
        var_type = parts[1]
        info_map = {}
        s = parts[2].strip('{}')
        #print(s)
        parts1 = s.split(',')
        for ss in parts1:
            s1 = ss.split(':')[0].strip(' ').strip('\'')
            s2 = ss.split(':')[1].strip(' ').strip('\'')
            if var_type=='CONTINUOUS':
                info_map[s1] = float(s2)
            else:
                info_map[int(s1)] = float(s2)
        quant_info.append((var_name,var_type,info_map))
    return quant_info

def invert_quant_info(quant_info):
    dequant_info = []
    for qi in quant_info:
        qmap = qi[2]
        dqmap = {}
        if qi[1]=='CONTINUOUS':
            dqmap = qmap
        else:
            for k in qmap:
                dqmap[qmap[k]] = k
        dequant_info.append((qi[0],qi[1],dqmap))
    return dequant_info

def quantify_nans(df,file_to_write=''):
    # quantifying the nans
    var_missing_val = df.isnull().sum()
    if not file_to_write=='':
        var_missing_val.to_csv('var_missing_val.txt', index=True,sep='\t')

    var_missing_sample = df.transpose().isnull().sum()
    if not file_to_write=='':
        var_missing_sample.to_csv('var_missing_sample.txt', index=True,sep='\t')

    print('Missing values {} ({}%)'.format(df.isnull().sum().sum(),100*df.isnull().sum().sum()/(df.shape[1]-1)/(df.shape[0])))

    # extract the complete part of the dataset

    dfc = df.dropna()

    if not file_to_write=='':
        dfc.to_csv(file_to_write,index=False,sep='\t')

    print('Number of complete rows: {} ({}%)'.format(dfc.shape[0],100*dfc.shape[0]/df.shape[0]))


def detect_variable_type(df,Max_Number_Of_Ordinal_Values=10,verbose=True):
    # classification of variables into continuous, binary, ordinal
    # let us assume for the moment that all categorical variables have been converted to the (transformed?) complete disjunctive table
    # hence they will be represented as sets of binary variables
    
    binary = []
    continuous = []
    ordinal = []
    variable_types = []

    for col in df.columns[1:]: 
        vals = np.sort(df[col].unique())
        #print(col,vals)
        vals = [x for x in vals if str(x) != 'nan']
        tp = 'UNKNOWN'
        # NaNs must be 
        if len(vals)==2 or len(vals)==1:
            tp = 'BINARY'
            binary.append(col)
        if len(vals)>Max_Number_Of_Ordinal_Values:
            tp = 'CONTINUOUS'
            continuous.append(col)
        if len(vals)>2 and len(vals)<=Max_Number_Of_Ordinal_Values:
            tp = 'ORDINAL'
            ordinal.append(col)
        variable_types.append(tp)
        if len(vals)>Max_Number_Of_Ordinal_Values:
            if verbose:
                print(col,'\t','\t[',np.min(vals),'... ',len(vals),'values...',np.max(vals),']\t',tp)
        else:
            if verbose:
                print(col,'\t',vals,'\t',len(vals),'\t',tp)
    return variable_types, binary, continuous, ordinal

def correct_column_types_in_dataframe(df,variable_types):
    for i,col in enumerate(df.columns[1:]):
        if variable_types[i]=='BINARY' or variable_types[i]=='ORDINAL':
            df = df.astype({col: 'int32'})

def SVDcomplete_imputation_method(dfq,variable_types,num_components=-1,produce_plots=True,verbose=True):

    # IMPUTATION METHOD 'SVDcomplete'
    # dfq - quantified dataframe
    # imputation via computing SVD on complete matrix and then projecting vectors with missing values 
    # on the first num_components principal components
    # the only parameter of imputation - num_components - how many pca components to use
    # reasonable choice is the linear effective dimensionality

    dframe = dfq.dropna()
    dframe_mv = dfq
    # we skip the first column - it is assumed to be the patient ids
    X = dframe[dframe.columns[1:]].to_numpy()

    if verbose:
        print('Matrix shape:',X.shape)
    
    # making pca on the complete matrix

    if num_components<0:
        pca = PCA(n_components=X.shape[1])
        u = pca.fit_transform(X)
        v = pca.components_.T
        s = pca.explained_variance_ratio_
        sn = s/s[0]
        lin_dim = len(np.where(sn>0.1)[0])
        if verbose:
            print('Effective linear dimension',lin_dim)
        num_components = lin_dim
        
    pca = PCA(n_components=num_components)
    u = pca.fit_transform(X)
    v = pca.components_.T
    s = pca.explained_variance_ratio_
        
    if produce_plots:
        plt.figure(figsize=(10,15))
        plt.subplot(321)
        plt.plot(s,'ko-')
        plt.xlabel('Principal component')
        plt.ylabel('Explained variance')
        plt.subplot(322)
        plt.plot(u[:,0],u[:,1],'ko',markersize=10)
        plt.xlabel('PC1 {}%'.format(100*s[0]))
        plt.ylabel('PC2 {}%'.format(100*s[1]))

    # Extracting the matrix with missing values
    X_mv = dframe_mv[dframe_mv.columns[1:]].to_numpy()
    if verbose:
        print('Full matrix shape',X_mv.shape)

    # Centering the matrix with missing values

    # This piece of code is not needed, left just in case
    #mean_matrix_mv = np.zeros((X_mv.shape[0],X_mv.shape[1]))
    #for i in range(0,X_mv.shape[1]):
    #    vr = X_mv[:,i]
    #    mv = np.mean(vr[~np.isnan(vr)])
    #    mean_matrix_mv[:,i] = mv
    #Xmvc = X_mv - mean_matrix_mv

    Xmvc = X_mv - X.mean(axis=0,keepdims=True)

    # Projecting vectors with missing values on PC vectors
    u_mv = np.zeros((Xmvc.shape[0],v.shape[1]))
    for i in range(0,Xmvc.shape[0]):
        xi = Xmvc[i,:]
        #if np.isnan(xi).sum()>0:
        #    print('x',str(i),'=',xi)
        for j in range(0,v.shape[1]):
            vj = v[:,j]
            u_mv[i,j] = np.nansum(xi*vj)
        #if np.isnan(xi).sum()>0:
        #    print('u_mv[i]=',u_mv[i,:])

    
    # Some visualization of where the vectors with missing values were projected
    mv = np.isnan(Xmvc).sum(axis=1)

    if produce_plots:
        inds = np.where(mv>0)
        x = np.transpose(u_mv[inds,0])
        y = np.transpose(u_mv[inds,1])
        plt.plot(x,y,'ro')
        inds = np.where(mv==0)
        x = np.transpose(u_mv[inds,0])
        y = np.transpose(u_mv[inds,1])
        plt.plot(x,y,'mo')
        plt.legend(['full','missing','full_reproj'])

    # Injecting the manifold back into the data space
    Xmvc_proj = np.transpose(np.matmul(v,np.transpose(u_mv)))+X.mean(axis=0,keepdims=True)
    
    #print('v',v)

    # Performing pca on the injected manifold, just in case for check
    pca1 = PCA(n_components=num_components)
    u1 = pca1.fit_transform(Xmvc_proj)
    v1 = pca1.components_.T
    s1 = pca1.explained_variance_ratio_
    
    if produce_plots:
        plt.subplot(323)
        plt.plot(s1,'ko-')
        plt.xlabel('Principal component')
        plt.ylabel('Explained variance')
        plt.subplot(324)
        plt.plot(u1[:,0],u1[:,1],'ko',markersize=10)
        plt.xlabel('PC1 {}%'.format(100*s1[0]))
        plt.ylabel('PC2 {}%'.format(100*s1[1]))

    # Now performing the actual value imputing. 
    # In case of ordinal and binary variables, we round each imputed 
    # value to the closest discrete value, otherwise we use the imputed value directly

    dfq_imputed = dframe_mv.copy()

    for i in range(1,dframe_mv.shape[1]):
        var_values = X_mv[:,i-1]
        var_values = var_values[np.where(~np.isnan(var_values))]
        var_values = np.unique(var_values)
        for j in range(0,dframe_mv.shape[0]):
            val = dframe_mv.loc[j,dframe_mv.columns[i]]
            if np.isnan(val):
                val_imputed = Xmvc_proj[j,i-1]
                #print('Imputed',str(j),str(i-1),str(val_imputed))
                diffs = np.abs(var_values-val_imputed)
                if variable_types[i-1]=='ORDINAL' or variable_types[i-1]=='BINARY':
                    val_imputed = var_values[np.argmin(diffs)]
                #print(val_imputed,var_values,diffs)
                dfq_imputed.loc[j,dfq_imputed.columns[i]] = val_imputed

    # Performing pca on the imputed dataset, just in case for check

    X = dfq_imputed[dfq_imputed.columns[1:]].to_numpy()
    pca2 = PCA(n_components=num_components)
    u2 = pca2.fit_transform(X)
    v2 = pca2.components_.T
    s2 = pca2.explained_variance_ratio_
    
    if produce_plots:
        plt.subplot(325)
        plt.plot(s2,'ko-')
        plt.xlabel('Principal component')
        plt.ylabel('Explained variance')
        plt.subplot(326)
        plt.plot(u2[:,0],u1[:,1],'ko',markersize=10)
        plt.xlabel('PC1 {}%'.format(100*s2[0]))
        plt.ylabel('PC2 {}%'.format(100*s2[1]))
    
    return dfq_imputed


def convert_categorical_variables_and_save(file_name, columns_to_encode, verbose=True):
    df = pd.read_csv(file,delimiter='\t')
    if verbose:
        display(df)
    df_dummies = pd.get_dummies(df,columns=columns_to_encode)
    if verbose:
        display(df_dummies)
    df_dummies.to_csv(file[:-4]+'_dummies.txt', index=False,sep='\t')
