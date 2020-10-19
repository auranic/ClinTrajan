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
import scipy.stats
from clintraj_qi import detect_variable_type
import matplotlib.pyplot as plt
import seaborn as sns

def pairwise_corr_sum_Q2(Y):
    # returns Q2 matrix - optimization criteria - sum of pairwise Pearson correlations
    cr = np.corrcoef(Y.T)
    cr2 = cr*cr
    return (np.sum(cr2)-cr.shape[0])/2

def split_into_continuos_ordinal(X,var_types):
    # split into continuous and ordinal parts
    indc = [i for i, x in enumerate(var_types) if x == 'CONTINUOUS']
    indo = [i for i, x in enumerate(var_types) if x == 'ORDINAL']
    Xc = X[:,indc]
    Xo = X[:,indo]
    return Xc,Xo,indc,indo

def extract_quantification_table(Xo):
    cik = [np.unique(Xo[:,i]) for i in range(Xo.shape[1])]
    return cik


def gradient_Q2(X,var_types):
    # compute the gradient for a matrix containing mix of CONTINUOUS, 
    # ORDINAL and BINARY variables (as defined in var_types)
    # returns the gradient for adjusting each catergory value for an ordinal value
    
    Xc,Xo,indc,indo = split_into_continuos_ordinal(X,var_types)
    Npoints = Xo.shape[0]
    
    # quantifying ordinal variables, computing cik, Nco, pi
    cik = extract_quantification_table(Xo)
        
    Nco = {}
    for i in range(Xo.shape[1]):
        for j in range(Xo.shape[1]):
            li = cik[i].shape[0]
            lj = cik[j].shape[0]            
            cooc_ij = np.zeros((li,lj))
            for k in range(Npoints):
                ind_i = np.where(cik[i]==Xo[k,i])[0][0]
                ind_j = np.where(cik[j]==Xo[k,j])[0][0]
                cooc_ij[ind_i,ind_j]=cooc_ij[ind_i,ind_j]+1
            Nco[(i,j)] = cooc_ij
            #Nco[(j,i)] = cooc_ij.T
    pi = []
    for i in range(Xo.shape[1]):
        pi.append(np.diag(Nco[(i,i)])/Npoints)

    # quantifying continuous variables, computes c_av_jik
    c_av_jik = []
    for j in range(Xc.shape[1]):
        c_av_ik = []
        for i in range(Xo.shape[1]):
            c_av = np.zeros((cik[i].shape[0]))
            for k,val in enumerate(cik[i]):
                inds = np.where(Xo[:,i]==val)
                #print(val,':',inds)
                c_av[k] = np.mean(Xc[inds,j])
            c_av_ik.append(c_av)
        c_av_jik.append(c_av_ik)
        
    # compute gradient
    a_i_kj = []
    for i in range(Xo.shape[1]):
        li = cik[i].shape[0]
        akj = np.zeros((li,li))
        for k in range(li):
            for j in range(li):
                for m in range(Xo.shape[1]):
                    if not m==i:
                        lm = cik[m].shape[0]
                        for i1 in range(lm):
                            for i2 in range(lm):
                                akj[k,j] = akj[k,j] + Nco[(i,m)][k,i1]*Nco[(m,i)][i2,j]*cik[m][i1]*cik[m][i2]
                for m in range(Xc.shape[1]):
                    akj[k,j] = akj[k,j] + c_av_jik[m][i][k]*c_av_jik[m][i][j]*pi[i][j]*pi[i][k]
        a_i_kj.append(akj)
    grad_cik = []
    for i in range(Xo.shape[1]):
        li = cik[i].shape[0]
        gk = np.zeros(li)
        for k in range(li):
            for j in range(li):
                gk[k]=gk[k]+a_i_kj[i][k,j]*cik[i][j]
        grad_cik.append(gk)
    return grad_cik

def update_quantification_table(cik,grad_cik,learning_step=None,fraction_max_step=0.9):
    # takes the old quantification table and updates it using the gradient,
    # the gradient step is automatically evaluated if not specified
    cik_new = []
    if learning_step is None:
        learning_step = estimate_learning_step(cik,grad_cik,fraction_max_step=fraction_max_step)
        cik_new,learning_step = update_quantification_table(cik,grad_cik,learning_step)
        #print('Learning_step',learning_step)
        while not is_quantification_table_monotonic(cik_new)[0]:
            learning_step=learning_step/2
            #print('Learning_step',learning_step)
            cik_new,learning_step = update_quantification_table(cik,grad_cik,learning_step)
    else:
        for i in range(len(cik)):
            li = cik[i].shape[0]
            ci = cik[i]+grad_cik[i]*learning_step
            cik_new.append(ci)
    return cik_new,learning_step
    

def estimate_learning_step(cik,grad_cik,fraction_max_step=0.9):
    estimated_step = 100000
    for i in range(len(cik)):
        li = cik[i].shape[0]
        for j in range(li):
            coord = cik[i][j]
            grad_coord = grad_cik[i][j]
            if abs(grad_coord)>1e-10:
                if grad_coord>0:
                    if j>0:
                        max_step = (cik[i][j]-cik[i][j-1])/grad_coord
                        if max_step<estimated_step:
                            estimated_step=max_step
                else:
                    if j<li-1:
                        max_step = (cik[i][j]-cik[i][j+1])/grad_coord
                        if max_step<estimated_step:
                            estimated_step=max_step
    return estimated_step*fraction_max_step
    

def update_matrix_with_new_quantification(X,var_types, cik_new,verbose=True):
    # replace values in the data matrix for ordinal variables with new quantification table
    X_new = X.copy()
    i = 0
    _,Xo,_,_ = split_into_continuos_ordinal(X,var_types)
    cik = extract_quantification_table(Xo)
    for k in range(len(var_types)):
        if var_types[k]=='ORDINAL':
            vals = cik[i]
            for j,val in enumerate(vals):
                X_new[:,k] = np.where(X_new[:,k]==val, cik_new[i][j], X_new[:,k]) 
            i=i+1
    if verbose:
        print('Difference between X and X_new:',np.sum(X-X_new)*np.sum(X-X_new))   
    return X_new

def is_quantification_table_monotonic(cik):
    delta=10000
    monotonic = True
    for x in cik:
        if not strictly_increasing(x):
            monotonic = False
        for k in range(len(x)-1):
            dx = x[k+1]-x[k]
            if dx<delta:
                delta=dx
    return monotonic,delta
        
def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def optimal_scaling(X,var_types,verbose=False,producePlots=True,max_num_iterations = 100,tolerance=1e-5,vmax=1,fraction_max_step=0.9):
    Q2 = pairwise_corr_sum_Q2(X)
    if(verbose):
        print('\n','Initial Q2:',Q2)
        print('Initial quantification table:')
    Xc,Xo,indc,indo = split_into_continuos_ordinal(X,var_types)
    cik_initial = extract_quantification_table(Xo)
    if(verbose):
        print(cik_initial)
        print('Monotonicity:',is_quantification_table_monotonic(cik_initial))

    Q2_vals = []
    Learning_rate = [] 
    Diff = []
    Monotonicity_Deltas = []
    X0 = X.copy()

    cik = cik_initial
    cik_complete = cik_initial

    for iteration in range(max_num_iterations):
        if(verbose):
            print('Iteration',iteration,'Q2=',Q2)
        X = scipy.stats.zscore(X)
        Xc,Xo,indc,indo = split_into_continuos_ordinal(X,var_types)
        cik_old = cik
        #print('Extract quantification....')
        cik = extract_quantification_table(Xo)
        #print('Compute Q2....')
        Q2 = pairwise_corr_sum_Q2(X)
        Q2_vals.append(Q2)
        #print('Compute gradient...')
        grad_cik = gradient_Q2(X,var_types)
        #print('Gradient:',grad_cik)
        #print('Update the quantification...')
        cik_new,learning_step = update_quantification_table(cik,grad_cik,fraction_max_step=fraction_max_step)
        Learning_rate.append(learning_step)
        #print('Checking monotonicity...')
        monotone,delta = is_quantification_table_monotonic(cik_new)
        Monotonicity_Deltas.append(delta)
        #print('Update data matrix...')
        X_new = update_matrix_with_new_quantification(X,var_types, cik_new,verbose=False)
        diff = np.sum(X-X_new)*np.sum(X-X_new)/np.sum(X)*np.sum(X)
        Diff.append(diff)
        X = X_new
        cik = cik_new
        if diff<tolerance:
            break;
    Q2_vals = np.array(Q2_vals)
    Learning_rate = np.array(Learning_rate)

    if producePlots:
        fig = plt.figure(figsize=(20,10))
        plt.subplot(231)
        plt.plot(Q2_vals,'bo-')
        plt.title('Q2')
        plt.subplot(232)
        plt.plot(Learning_rate,'ro-')
        plt.title('Learning rate')
        plt.subplot(233)
        plt.plot(Diff,'go-')
        plt.title('Difference in X')

        plt.subplot(234)
        plt.plot(Monotonicity_Deltas,'mo-')
        plt.title('Delta')

        plt.subplot(235)
        inds_co = indc+indo
        cr = np.corrcoef(X0[:,inds_co].T)
        #cr = scipy.stats.spearmanr(X0).correlation
        cr2 = cr*cr
        sns.heatmap(cr2, vmin=0, vmax=vmax)
        plt.title('Initial Q2 table')
        plt.subplot(236)
        cr = np.corrcoef(X[:,inds_co].T)
        #cr = scipy.stats.spearmanr(X).correlation
        cr2 = cr*cr
        sns.heatmap(cr2, vmin=0, vmax=vmax)
        plt.title('Final Q2 table')

        X = scipy.stats.zscore(X)
        Xc,Xo,indc,indo = split_into_continuos_ordinal(X,var_types)
        cik = extract_quantification_table(Xo)
        
    if verbose:
        print('\n','Q2 after update:',pairwise_corr_sum_Q2(X))
        print('Final quantification table:')
        print(cik)
        print('Monotonicity:',is_quantification_table_monotonic(cik))
        
    return X,cik
