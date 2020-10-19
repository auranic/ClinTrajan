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


import trimap
import getpass
import os
from os import path


import random
import numpy as np
import matplotlib.pyplot as plt
from time import time
from matplotlib.ticker import NullFormatter
import pandas as pd

from sklearn import manifold, datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from umap import UMAP

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K

def apply_panel_of_manifold_learning_methods(X,color,
                                Color_by_branches=[],precomputed_results={},
                                color_map='cool',ColorByFeature='',
                                variable_names=[],ElMapFolder='',
                                n_neighbors=20, n_components = 2,
                                title_fontsize = 30,points_size = 30,
                                methods_to_apply=[],
                                n_subplots_x = 4, n_subplots_y = 3,
                                figsizex = 20, figsizey =20):
    viz_results = precomputed_results
    #Set figure parameters
    n_points = X.shape[0]
    
    #cmap = plt.cm.Paired
    #cmap = 'hot'
    cmap = color_map
    # cmap = plt.cm.tab20
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(figsizex, figsizey))
    applyAllMethods = True
    if len(methods_to_apply)>0:
        applyAllMethods = False

    color1 = color
    if len(Color_by_branches)>0:
        #color1 = vec_labels_by_branches
        color2 = Color_by_branches
        color_seq = [[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0],
             [1,0,0.5],[1,0.5,0],[0.5,0,1],[0.5,1,0],
             [0.5,0.5,1],[0.5,1,0.5],[1,0.5,0.5],
             [0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0],[0.5,0.5,0.5],[0,0,0.5],[0,0.5,0],[0.5,0,0],
             [0,0.25,0.5],[0,0.5,0.25],[0.25,0,0.5],[0.25,0.5,0],[0.5,0,0.25],[0.5,0.25,0],
             [0.25,0.25,0.5],[0.25,0.5,0.25],[0.5,0.25,0.25],[0.25,0.25,0.5],[0.25,0.5,0.25],
             [0.25,0.25,0.5],[0.25,0.5,0.25],[0.5,0,0.25],[0.5,0.25,0.25]]
        color2_unique, color2_count = np.unique(color2, return_counts=True)
        inds = sorted(range(len(color2_count)), key=lambda k: color2_count[k], reverse=True)
        newc = []
        for i,c in enumerate(color2):
            k = np.where(color2_unique==c)[0][0]
            count = color2_count[k]
            k1 = np.where(inds==k)[0][0]
            k1 = k1%len(color_seq)
            col = color_seq[k1]
            newc.append(col)
        color2 = newc
        color1 = color2

    if not ColorByFeature=='':
        k = variable_names.index(ColorByFeature)
        #color1 = X_original[:,k]
        color1 = X[:,k]

    onlyDraw = not len(precomputed_results)==0

    print('Start computations...')

    # some standard methods
    i = 1

    ####################### PCA #########################
    if applyAllMethods or 'PCA' in methods_to_apply:
        pca = PCA(n_components=n_components)
        t0 = time()
        if  not onlyDraw or not 'PCA' in precomputed_results:
            Y_PCA = pca.fit_transform(X)
            viz_results['PCA'] = Y_PCA
        else:
            Y_PCA = precomputed_results['PCA']
        t1 = time()
        print("PCA: %.2g sec" % (t1 - t0))

        ax = fig.add_subplot(n_subplots_x,n_subplots_y,i)
        plt.scatter(Y_PCA[:, 0], Y_PCA[:, 1], c=color1, cmap=cmap,s=points_size)
        plt.title("PCA",fontdict = {'fontsize' : title_fontsize})
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')


    ### LLE ###
    if applyAllMethods or 'LLE' in methods_to_apply:
        t0 = time()
        if  not onlyDraw or not 'LLE' in precomputed_results:
            print('Computing LLE...')
            Y_LLE = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                    eigen_solver='auto',
                                    method='standard').fit_transform(X)
            viz_results['LLE'] = Y_LLE
        else:
             Y_LLE = viz_results['LLE']
        t1 = time()
        print("%s: %.2g sec" % ('LLE', t1 - t0))
        i+=1
        ax = fig.add_subplot(n_subplots_x,n_subplots_y,i)
        plt.scatter(Y_LLE[:, 0], Y_LLE[:, 1], c=color1, cmap=cmap,s=points_size)
        plt.title("LLE",fontdict = {'fontsize' : title_fontsize})
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')


    ### Modified LLE ###
    if applyAllMethods or 'MLLE' in methods_to_apply:
        t0 = time()
        if  not onlyDraw  or not 'MLLE' in precomputed_results:
            Y_MLLE = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                        eigen_solver='auto',
                                        method='modified').fit_transform(X)
            viz_results['MLLE'] = Y_MLLE
        else:
            Y_MLLE = viz_results['MLLE']
        t1 = time()
        print("%s: %.2g sec" % ('Modified LLE', t1 - t0))
        i+=1
        ax = fig.add_subplot(n_subplots_x,n_subplots_y,i)
        plt.scatter(Y_MLLE[:, 0], Y_MLLE[:, 1], c=color1, cmap=cmap,s=points_size)
        plt.title("MLLE",fontdict = {'fontsize' : title_fontsize})
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')


    ### ISOMAP ###
    if applyAllMethods or 'ISOMAP' in methods_to_apply:
        i += 1
        t0 = time()
        if  not onlyDraw or not 'ISOMAP' in precomputed_results:
            Y_ISOMAP = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
            viz_results['ISOMAP'] = Y_ISOMAP
        else:
            Y_ISOMAP = viz_results['ISOMAP']
        t1 = time()
        print("Isomap: %.2g sec" % (t1 - t0))
        ax = fig.add_subplot(n_subplots_x,n_subplots_y,i)
        plt.scatter(Y_ISOMAP[:, 0], Y_ISOMAP[:, 1], c=color1, cmap=cmap,s=points_size)
        plt.title("Isomap",fontdict = {'fontsize' : title_fontsize})
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')


    ### MDS ###
    if applyAllMethods or 'MDS' in methods_to_apply:    
        i += 1
        t0 = time()
        if  not onlyDraw or not 'MDS' in precomputed_results:
            mds = manifold.MDS(n_components, max_iter=100, n_init=1)
            Y_MDS = mds.fit_transform(X)
            viz_results['MDS'] = Y_MDS
        else:
            Y_MDS = viz_results['MDS']
        t1 = time()
        print("MDS: %.2g sec" % (t1 - t0))
        ax = fig.add_subplot(n_subplots_x,n_subplots_y,i)
        plt.scatter(Y_MDS[:, 0], Y_MDS[:, 1], c=color1, cmap=cmap,s=points_size)
        plt.title("MDS",fontdict = {'fontsize' : title_fontsize})
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    ### SpectralEmbedding ###
    if applyAllMethods or 'SE' in methods_to_apply:      
        i += 1
        t0 = time()
        if  not onlyDraw  or not 'SE' in precomputed_results:
            se = manifold.SpectralEmbedding(n_components=n_components,n_neighbors=n_neighbors)
            Y_se = se.fit_transform(X)
            viz_results['SE'] = Y_se
        else:
            Y_se = viz_results['SE']
        t1 = time()
        print("SpectralEmbedding: %.2g sec" % (t1 - t0))
        ax = fig.add_subplot(n_subplots_x,n_subplots_y,i)
        plt.scatter(Y_se[:, 0], Y_se[:, 1], c=color1, cmap=cmap,s=points_size)
        plt.title("SpectralEmbedding",fontdict = {'fontsize' : title_fontsize})
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')


    ### t-SNE ###
    if applyAllMethods or 'TSNE' in methods_to_apply:      
        i += 1
        t0 = time()
        if  not onlyDraw  or not 'TSNE' in precomputed_results:
            tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0, perplexity=100)
            Y_TSNE = tsne.fit_transform(X)
            viz_results['TSNE'] = Y_TSNE
        else:
            Y_TSNE = viz_results['TSNE']
        t1 = time()
        print("t-SNE: %.2g sec" % (t1 - t0))
        ax = fig.add_subplot(n_subplots_x,n_subplots_y,i)
        plt.scatter(Y_TSNE[:, 0], Y_TSNE[:, 1], c=color1, cmap=cmap,s=points_size)
        plt.title("t-SNE",fontdict = {'fontsize' : title_fontsize})
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')


    ### UMAP ###
    if applyAllMethods or 'UMAP' in methods_to_apply:          
        i += 1
        t0 = time()
        if  not onlyDraw  or not 'UMAP' in precomputed_results:
            um = UMAP(n_neighbors=n_neighbors,
                  n_components=n_components)
            Y_UMAP = um.fit_transform(X)
            viz_results['UMAP'] = Y_UMAP
        else:
            Y_UMAP = viz_results['UMAP']
        t1 = time()
        print("UMAP: %.2g sec" % (t1 - t0))
        ax = fig.add_subplot(n_subplots_x,n_subplots_y,i)
        plt.scatter(Y_UMAP[:, 0], Y_UMAP[:, 1], c=color1, cmap=cmap,s=points_size)
        plt.title("UMAP",fontdict = {'fontsize' : title_fontsize})
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    ### TRIMAP ###
    if applyAllMethods or 'TRIMAP' in methods_to_apply:              
        t0 = time()
        if  not onlyDraw  or not 'TRIMAP' in precomputed_results:
            Y_TRIMAP = trimap.TRIMAP(verbose=False).fit_transform(X)
            viz_results['TRIMAP'] = Y_TRIMAP
        else:
            Y_TRIMAP = viz_results['TRIMAP']
        t1 = time()
        print("TRIMAP: %.2g sec" % (t1 - t0))
        i += 1
        ax = fig.add_subplot(n_subplots_x,n_subplots_y,i)
        plt.scatter(Y_TRIMAP[:, 0], Y_TRIMAP[:, 1], c=color1, cmap=cmap,s=points_size)
        plt.title("TRIMAP",fontdict = {'fontsize' : title_fontsize})
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')


    ### ELMAP ###
    #if  not onlyDraw:
    if applyAllMethods or 'ELMAP' in methods_to_apply:                  
        if os.path.exists(ElMapFolder+'/tests/_elmap_proj.txt'):
            Xproj_pd = pd.read_csv(ElMapFolder+'/tests/_elmap_proj.txt', sep='\t',header=None)
            Xproj = Xproj_pd.to_numpy()[:,0:-1]
            Y_ELMAP = Xproj
            i += 1
            ax = fig.add_subplot(n_subplots_x,n_subplots_y,i)
            plt.scatter(Y_ELMAP[:, 0], Y_ELMAP[:, 1], c=color1, cmap=cmap,s=points_size)
            plt.title("ELMAP",fontdict = {'fontsize' : title_fontsize})
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')


    ### Autoencoder ###
    if applyAllMethods or 'AUTOENCODER' in methods_to_apply:        
        layer_sizes = [64,32,16,8]
        #encoder
        inputs = Input(shape=(X.shape[1],), name='encoder_input')
        x = inputs
        for size in layer_sizes:
            x = Dense(size, activation='relu',kernel_initializer='he_uniform')(x)
        latent = Dense(n_components,kernel_initializer='he_uniform', name='latent_vector')(x)
        encoder = Model(inputs, latent, name='encoder')

        #decoder
        latent_inputs = Input(shape=(n_components,), name='decoder_input')
        x = latent_inputs
        for size in layer_sizes[::-1]:
            x = Dense(size, activation='relu',kernel_initializer='he_uniform')(x)
        outputs = Dense(X.shape[1] ,activation='sigmoid',kernel_initializer='he_uniform',name='decoder_output')(x)
        decoder = Model(latent_inputs, outputs, name='decoder')

        #autoencoder
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')

        #model summary
        # encoder.summary()
        # decoder.summary()
        # autoencoder.summary()
        X_01 = (X-X.min())/(X.max()-X.min())
        autoencoder.compile(loss='mse', optimizer='adam')
        t0 = time()
        if  not onlyDraw  or not 'AUTOENCODER' in precomputed_results:
            autoencoder.fit(x=X_01,y=X_01,epochs=200,verbose=0)
            Y_AUTOENCODER = encoder.predict(X)
            viz_results['AUTOENCODER'] = Y_AUTOENCODER
        else:
            Y_AUTOENCODER = viz_results['AUTOENCODER']
        t1 = time()
        print("Autoencoder: %.2g sec" % (t1 - t0))

        i += 1
        ax = fig.add_subplot(n_subplots_x,n_subplots_y,i)
        plt.scatter(Y_AUTOENCODER[:, 0], Y_AUTOENCODER[:, 1], c=color1, cmap=cmap,s=points_size)
        plt.title("Autoencoder",fontdict = {'fontsize' : title_fontsize})
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')


    ### VAE ###
    if applyAllMethods or 'VAE' in methods_to_apply:        
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(n_components,))
            return z_mean + K.exp(z_log_var) * epsilon

        layer_sizes = [64,32,16,8]
        #encoder
        inputs = Input(shape=(X.shape[1],), name='encoder_input')
        x = inputs
        for size in layer_sizes:
            x = Dense(size, activation='relu',kernel_initializer='he_uniform')(x)

        z_mean = Dense(n_components,kernel_initializer='he_uniform', name='latent_mean')(x)
        z_log_var = Dense(n_components,kernel_initializer='he_uniform', name='latent_sigma')(x)

        z = Lambda(sampling, output_shape=(n_components,))([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        #decoder
        latent_inputs = Input(shape=(n_components,), name='decoder_input_sampling')
        x = latent_inputs
        for size in layer_sizes[::-1]:
            x = Dense(size, activation='relu',kernel_initializer='he_uniform')(x)
        outputs = Dense(X.shape[1] ,activation='sigmoid',kernel_initializer='he_uniform',name='decoder_output')(x)
        decoder = Model(latent_inputs, outputs, name='decoder')

        #autoencoder
        vae = Model(inputs, decoder(encoder(inputs)[2]), name='vae')

        def vae_loss(x, x_decoded_mean):
            xent_loss = K.mean(K.square((x- x_decoded_mean)))
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss
        vae.compile(optimizer='adam', loss=vae_loss)

        X_01 = (X-X.min())/(X.max()-X.min())
        t0 = time()
        if  not onlyDraw  or not 'VAE' in precomputed_results:
            vae.fit(x=X_01,y=X_01,epochs=200,verbose=0)
            Y_VAE = encoder.predict(X)[0]
            viz_results['VAE'] = Y_VAE
        else:
            Y_VAE = viz_results['VAE']
        t1 = time()
        print("VAE: %.2g sec" % (t1 - t0))
        i += 1
        ax = fig.add_subplot(n_subplots_x,n_subplots_y,i)
        plt.scatter(Y_VAE[:, 0], Y_VAE[:, 1], c=color1, cmap=cmap)
        plt.title("VAE",fontdict = {'fontsize' : title_fontsize})
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    plt.tight_layout()
    
    return viz_results
