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

import networkx as nx
import igraph
import scipy.stats
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import elpigraph
import gbrancher
from gbrancher import branch_labler
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math

from clintraj_util import moving_weighted_average
from clintraj_util import fill_gaps_in_number_sequence

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import scipy.optimize
import copy
import warnings
from elpigraph.src.graphs import ConstructGraph, GetSubGraph, GetBranches
from elpigraph.src.core import PartitionData
from elpigraph.src.distutils import PartialDistance
from elpigraph.src.reporting import project_point_onto_graph, project_point_onto_edge

from scipy import signal
from scipy.stats import spearmanr
import seaborn as sns 


def visualize_eltree_with_data(tree_elpi,X,X_original,principal_component_vectors,mean_vector,color,variable_names,
                              showEdgeNumbers=False,showNodeNumbers=False,showBranchNumbers=False,showPointNumbers=False,
                              Color_by_feature = '', Feature_Edge_Width = '', Invert_Edge_Value = False,
                              Min_Edge_Width = 5, Max_Edge_Width = 5, 
                              Big_Point_Size = 100, Small_Point_Size = 1, Normal_Point_Size = 20,
                              Visualize_Edge_Width_AsNodeCoordinates=True,
                              Color_by_partitioning = False,
                              visualize_partition = [],
                              Transparency_Alpha = 0.2,
                              Transparency_Alpha_points = 1,
                              verbose=False,
                              Visualize_Branch_Class_Associations = [], #list_of_branch_class_associations
                              cmap = 'cool',scatter_parameter=0.03,highlight_subset=[],
                              add_color_bar=False,
                              vmin=-1,vmax=-1,
                              percentile_contraction=20):
    """
       Function used to create a 2D visualization of the principal tree
       together with the data.
       Parameters
       ----------
       tree_elpi : ElPiGraph object
              the principal tree		
       X : ndarray
              the pre-processed data matrix which was used to construct the tree
              usually, it is a projection in the first principal components
       X_original : ndarray
              the initial data matrix containing all variables unpreprocessed
              the principal tree		
       principal_component_vectors : ndarray
              vectors of principal components
       mean_vector : ndarray
       color : ndarray 
              array of color values to be used in the scatter plotting
       variable_names : list of strings
              names of the variables in the same order as in X_original
    """
    nodep = tree_elpi['NodePositions']
    nodep_original = np.matmul(nodep,principal_component_vectors[:,0:X.shape[1]].T)+mean_vector
    adjmat = tree_elpi['ElasticMatrix']
    edges = tree_elpi['Edges'][0]
    color2 = color
    if not Color_by_feature=='':
        k = variable_names.index(Color_by_feature)
        color2 = X_original[:,k]
    if Color_by_partitioning:
        color2 = visualize_partition
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
    
    plt.style.use('ggplot')
    points_size = Normal_Point_Size*np.ones(X_original.shape[0])
    if len(Visualize_Branch_Class_Associations)>0:
        points_size = Small_Point_Size*np.ones(X_original.shape[0])
        for assc in Visualize_Branch_Class_Associations:
            branch = assc[0]
            cls = assc[1]
            indices = [i for i, x in enumerate(color) if x == cls]
            #print(branch,cls,color,np.where(color==cls))
            points_size[indices] = Big_Point_Size

    node_size = 10
    #Associate each node with datapoints
    if verbose:
        print('Partitioning the data...')
    partition, dists = elpigraph.src.core.PartitionData(X = X, NodePositions = nodep, MaxBlockSize = 100000000, TrimmingRadius = np.inf,SquaredX = np.sum(X**2,axis=1,keepdims=1))
    #col_nodes = {node: color[np.where(partition==node)[0]] for node in np.unique(partition)}


    #Project points onto the graph
    if verbose:
        print('Projecting data points onto the graph...')
    ProjStruct = elpigraph.src.reporting.project_point_onto_graph(X = X,
                                     NodePositions = nodep,
                                     Edges = edges,
                                     Partition = partition)

    projval = ProjStruct['ProjectionValues']
    edgeid = (ProjStruct['EdgeID']).astype(int)
    X_proj = ProjStruct['X_projected']

    dist2proj = np.sum(np.square(X-X_proj),axis=1)
    shift = np.percentile(dist2proj,percentile_contraction)
    dist2proj = dist2proj-shift

    #Create graph
    if verbose:
        print('Producing graph layout...')
    g=nx.Graph()
    g.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(g,scale=2)
    #pos = nx.planar_layout(g)
    #pos = nx.spring_layout(g,scale=2)
    idx=np.array([pos[j] for j in range(len(pos))])

    #plt.figure(figsize=(16,16))
    if verbose:
        print('Calculating scatter aroung the tree...')
    x = np.zeros(len(X))
    y = np.zeros(len(X))
    for i in range(len(X)):
        # distance from edge
	# This is squared distance from a node
        #r = np.sqrt(dists[i])*scatter_parameter
	# This is squared distance from a projection (from edge),
	# even though the difference might be tiny
        r = 0
        if dist2proj[i]>0:
            r = np.sqrt(dist2proj[i])*scatter_parameter        

        #get node coordinates for this edge
        x_coos = np.concatenate((idx[edges[edgeid[i],0],[0]],idx[edges[edgeid[i],1],[0]]))
        y_coos = np.concatenate((idx[edges[edgeid[i],0],[1]],idx[edges[edgeid[i],1],[1]]))

        projected_on_edge = False
  
        if projval[i]<0:
            #project to 0% of the edge (first node)
            x_coo = x_coos[0] 
            y_coo = y_coos[0]
        elif projval[i]>1: 
            #project to 100% of the edge (second node)
            x_coo = x_coos[1]
            y_coo = y_coos[1]
        else:   
            #project to appropriate % of the edge
            x_coo = x_coos[0] + (x_coos[1]-x_coos[0])*projval[i]
            y_coo = y_coos[0] + (y_coos[1]-y_coos[0])*projval[i]
            projected_on_edge = True

        #if projected_on_edge:
        #     color2[i]=0
        #else:
        #     color2[i]=1    
        #random angle
        #alpha = 2 * np.pi * np.random.random()
        #random scatter to appropriate distance 
        #x[i] = r * np.cos(alpha) + x_coo
        #y[i] = r * np.sin(alpha) + y_coo
	# we rather position the point close to project and put
	# it at distance r orthogonally to the edge 
	# on a random side of the edge 
        # However, if projection was on a node then we scatter 
        # in random direction
        vex = x_coos[1]-x_coos[0]
        vey = y_coos[1]-y_coos[0]
        if not projected_on_edge:
            vex = np.random.random()-0.5
            vey = np.random.random()-0.5
        vn = np.sqrt(vex*vex+vey*vey)
        vex = vex/vn
        vey = vey/vn
        rsgn = random_sign()
        x[i] = x_coo+vey*r*rsgn
        y[i] = y_coo-vex*r*rsgn
    if vmin<0:
        vmin=min(color2)
    if vmax<0:
        vmax=max(color2)
    plt.scatter(x,y,c=color2,cmap=cmap,s=points_size, vmin=vmin, vmax=vmax,alpha=Transparency_Alpha_points)
    if showPointNumbers:
        for j in range(len(X)):
            plt.text(x[j],y[j],j)
    if len(highlight_subset)>0:
        color_subset = [color2[i] for i in highlight_subset]
        plt.scatter(x[highlight_subset],y[highlight_subset],c=color_subset,cmap=cmap,s=Big_Point_Size)
    if add_color_bar:
        plt.colorbar()


    #Scatter nodes
    tree_elpi['NodePositions2D'] = idx
    plt.scatter(idx[:,0],idx[:,1],s=node_size,c='black',alpha=.8)

    #Associate edge width to a feature
    edge_vals = [1]*len(edges)
    if not Feature_Edge_Width=='' and not Visualize_Edge_Width_AsNodeCoordinates:
        k = variable_names.index(Feature_Edge_Width)
        for j in range(len(edges)):
            vals = X_original[np.where(edgeid==j)[0],k]
            vals = (np.array(vals)-np.min(X_original[:,k]))/(np.max(X_original[:,k])-np.min(X_original[:,k]))
            edge_vals[j] = np.mean(vals)
        for j in range(len(edges)):
            if np.isnan(edge_vals[j]):
                e = edges[j]
                inds = [ei for ei,ed in enumerate(edges) if ed[0]==e[0] or ed[1]==e[0] or ed[0]==e[1] or ed[1]==e[1]]
                inds.remove(j)
                evals = np.array(edge_vals)[inds]
                #print(j,inds,evals,np.mean(evals))
                edge_vals[j] = np.mean(evals[~np.isnan(evals)])
        if Invert_Edge_Value:
            edge_vals = [1-v for v in edge_vals]

    if not Feature_Edge_Width=='' and Visualize_Edge_Width_AsNodeCoordinates:
        k = variable_names.index(Feature_Edge_Width)
        for j in range(len(edges)):
            e = edges[j]
            amp = np.max(nodep_original[:,k])-np.min(nodep_original[:,k])
            mn = np.min(nodep_original[:,k])
            v0 = (nodep_original[e[0],k]-mn)/amp
            v1 = (nodep_original[e[1],k]-mn)/amp
            #print(v0,v1)
            edge_vals[j] = (v0+v1)/2
        if Invert_Edge_Value:
            edge_vals = [1-v for v in edge_vals]
        
    #print(edge_vals)
            
    
    #Plot edges
    for j in range(len(edges)):
        x_coo = np.concatenate((idx[edges[j,0],[0]],idx[edges[j,1],[0]]))
        y_coo = np.concatenate((idx[edges[j,0],[1]],idx[edges[j,1],[1]]))
        plt.plot(x_coo,y_coo,c='k',linewidth=Min_Edge_Width+(Max_Edge_Width-Min_Edge_Width)*edge_vals[j],alpha=Transparency_Alpha)
        if showEdgeNumbers:
            plt.text((x_coo[0]+x_coo[1])/2,(y_coo[0]+y_coo[1])/2,j,FontSize=20,bbox=dict(facecolor='grey', alpha=0.5))

    if showBranchNumbers:
        branch_vals = list(set(visualize_partition))
        for i,val in enumerate(branch_vals):
            ind = visualize_partition==val
            xbm = np.mean(x[ind])
            ybm = np.mean(y[ind])
            plt.text(xbm,ybm,int(val),FontSize=20,bbox=dict(facecolor='grey', alpha=0.5))
        
    if showNodeNumbers:
        for i in range(nodep.shape[0]):
            plt.text(idx[i,0],idx[i,1],str(i),FontSize=20,bbox=dict(facecolor='grey', alpha=0.5))

    #plt.axis('off')

def convert_elpigraph_to_igraph(elpigraph):
    """
     Function converting ElPiGraph object into igraph object    
    """
    edges = elpigraph['Edges'][0]
    nodes_positions = elpigraph['NodePositions']
    g = igraph.Graph()
    g.add_vertices(len(nodes_positions))
    g.add_edges(edges)
    return g


def partition_data_by_tree_branches(X,tree_elpi):
    """
     Partitioning of data points from X accordingly to the segments 
     (tree branches) of the principal tree
    """
    edges = tree_elpi['Edges'][0]
    nodes_positions = tree_elpi['NodePositions']
    g = igraph.Graph()
    g.add_vertices(len(nodes_positions))
    g.add_edges(edges)
    vec_labels_by_branches = branch_labler( X , g, nodes_positions )
    return vec_labels_by_branches


def prune_the_tree(tree_elpi):
    """
     remove short 'orphan' edges not forming a continuous branch
    """
    old_tree = tree_elpi.copy()
    edges = tree_elpi['Edges'][0]
    nodes_positions = tree_elpi['NodePositions']

    g = igraph.Graph()
    g.add_vertices(len(nodes_positions ))
    labels = []
    for i in range(0,len(nodes_positions )):
        labels.append(i)
    g.vs['label'] = labels
    g.add_edges(edges)

    degs = g.degree()
    list_to_remove = []
    for e in g.get_edgelist():
        if degs[e[0]]==1 and degs[e[1]]>2:
            list_to_remove.append(e[0])
        if degs[e[1]]==1 and degs[e[0]]>2:
            list_to_remove.append(e[1])
    list_to_remove = list(set(list_to_remove))
    g.delete_vertices(list_to_remove)

    edge_array = np.zeros((len(g.get_edgelist()),2),'int32')
    for i,e in enumerate(g.get_edgelist()):
        edge_array[i,0] = e[0]
        edge_array[i,1] = e[1]

    print('Removed',len(list_to_remove),'vertices and',len(old_tree['Edges'][0])-len(edge_array),'edges')

    tree_elpi['Edges'] = (edge_array,tree_elpi['Edges'][1])
    vs_subset = g.vs['label']
    tree_elpi['NodePositions'] = nodes_positions[vs_subset,:]

def random_sign():
    """
     Gives -1 or 1 randomly
    """
    return 1 if random.random() < 0.5 else -1

def ExtendLeaves_modified(X, PG,
                Mode = "QuantDists",
                ControlPar = .9,
                DoSA = True,
                DoSA_maxiter = 2000,
                LeafIDs = None,
                TrimmingRadius = float('inf'),
                PlotSelected = False):
    '''
    #' Extend leaves with additional nodes
    #'
    #' @param X numeric matrix, the data matrix
    #' @param TargetPG list, the ElPiGraph structure to extend
    #' @param LeafIDs integer vector, the id of nodes to extend. If NULL, all the vertices will be extended.
    #' @param TrimmingRadius positive numeric, the trimming radius used to control distance 
    #' @param ControlPar positive numeric, the paramter used to control the contribution of the different data points
    #' @param DoSA bollean, should optimization (via simulated annealing) be performed when Mode = "QuantDists"?
    #' @param Mode string, the mode used to extend the graph. "QuantCentroid" and "WeigthedCentroid" are currently implemented
    #' @param PlotSelected boolean, should a diagnostic plot be visualized
    #'
    #' @return The extended ElPiGraph structure
    #'
    #' The value of ControlPar has a different interpretation depending on the valus of Mode. In each case, for only the extreme points,
    #' i.e., the points associated with the leaf node that do not have a projection on any edge are considered.
    #'
    #' If Mode = "QuantCentroid", for each leaf node, the extreme points are ordered by their distance from the node
    #' and the centroid of the points farther away than the ControlPar is returned.
    #'
    #' If Mode = "WeightedCentroid", for each leaf node, a weight is computed for each points by raising the distance to the ControlPar power.
    #' Hence, larger values of ControlPar result in a larger influence of points farther from the node
    #'
    #' If Mode = "QuantDists", for each leaf node, ... will write it later
    #'
    #'
    #' @export
    #'
    #' @examples
    #'
    #' TreeEPG <- computeElasticPrincipalTree(X = tree_data, NumNodes = 50,
    #' drawAccuracyComplexity = FALSE, drawEnergy = FALSE)
    #'
    #' ExtStruct <- ExtendLeaves(X = tree_data, TargetPG = TreeEPG[[1]], Mode = "QuantCentroid", ControlPar = .5)
    #' PlotPG(X = tree_data, TargetPG = ExtStruct)
    #'
    #' ExtStruct <- ExtendLeaves(X = tree_data, TargetPG = TreeEPG[[1]], Mode = "QuantCentroid", ControlPar = .9)
    #' PlotPG(X = tree_data, TargetPG = ExtStruct)
    #'
    #' ExtStruct <- ExtendLeaves(X = tree_data, TargetPG = TreeEPG[[1]], Mode = "WeigthedCentroid", ControlPar = .2)
    #' PlotPG(X = tree_data, TargetPG = ExtStruct)
    #'
    #' ExtStruct <- ExtendLeaves(X = tree_data, TargetPG = TreeEPG[[1]], Mode = "WeigthedCentroid", ControlPar = .8)
    #' PlotPG(X = tree_data, TargetPG = ExtStruct)
    #'
    '''

    TargetPG = copy.deepcopy(PG)
    # Generate net
    Net = ConstructGraph(PrintGraph = TargetPG)

    # get leafs
    if(LeafIDs is None):
        LeafIDs = np.where(np.array(Net.degree()) == 1)[0]

    # check LeafIDs
    if(np.any(np.array(Net.degree(LeafIDs)) > 1)):
        raise ValueError("Only leaf nodes can be extended")

    # and their neigh
    Nei = Net.neighborhood(LeafIDs,order = 1)

    # and put stuff together
    NeiVect = list(map(lambda x: set(Nei[x]).difference([LeafIDs[x]]),range(len(Nei))))
    NeiVect = np.array([j for i in NeiVect for j in list(i)])
    NodesMat = np.hstack((LeafIDs[:,None], NeiVect[:,None]))

    # project data on the nodes
    PD = PartitionData(X = X, NodePositions = TargetPG['NodePositions'], MaxBlockSize = 10000000, TrimmingRadius = TrimmingRadius, SquaredX=np.sum(X**2,axis=1,keepdims=1))

    # Keep track of the new nodes IDs
    NodeID = len(TargetPG['NodePositions'])-1

    init = False
    NNPos = None
    NEdgs = None
    UsedNodes = []

    # for each leaf
    for i in range(len(NodesMat)):

        if(np.sum(PD[0] == NodesMat[i,0]) == 0):
            continue

        # generate the new node id
        NodeID = NodeID + 1

        # get all the data associated with the leaf node
        tData = X[(PD[0] == NodesMat[i,0]).flatten(),:]

        # and project them on the edge
        Proj = project_point_onto_edge(X = X[(PD[0] == NodesMat[i,0]).flatten(),:],
                                                                        NodePositions = TargetPG['NodePositions'],
                                                                        Edge = NodesMat[i,:])

        # Select the distances of the associated points
        Dists = PD[1][PD[0] == NodesMat[i,0]]

        # Set distances of points projected on beyond the initial position of the edge to 0
        Dists[Proj['Projection_Value'] >= 0] = 0

        if(Mode == "QuantCentroid"):
            ThrDist = np.quantile(Dists[Dists>0], ControlPar)
            SelPoints = np.where(Dists >= ThrDist)[0]

            print(len(SelPoints), " points selected to compute the centroid while extending node", NodesMat[i,0])

            if(len(SelPoints)>1):
                NN = np.mean(tData[SelPoints,:],axis=0,keepdims=1)

            else :
                NN = tData[SelPoints,:]

            # Initialize the new nodes and edges
            if not init:
                init=True
                NNPos = NN.copy()
                NEdgs = np.array([[NodesMat[i,0], NodeID]])
                UsedNodes.extend(np.where(PD[0].flatten() == NodesMat[i,0])[0][SelPoints])
            else:
                NNPos = np.vstack((NNPos, NN))
                NEdgs = np.vstack((NEdgs, np.array([[NodesMat[i,0], NodeID]])))
                UsedNodes.extend(list(np.where(PD[0].flatten() == NodesMat[i,0])[0][SelPoints]))


        if(Mode == "WeightedCentroid"):
            Dist2 = Dists**(2*ControlPar)
            Wei = Dist2/np.max(Dist2)

            if(len(Wei)>1):
                NN = np.sum(tData*Wei[:,None],axis=0)/np.sum(Wei)

            else:
                NN = tData

            # Initialize the new nodes and edges
            if not init:
                init=True
                NNPos = NN.copy()
                NEdgs = np.array([NodesMat[i,0], NodeID])
                WeiVal = list(Wei)
                UsedNodes.extend(list(np.where(PD[0].flatten() == NodesMat[i,0])[0]))
            else:
                NNPos = np.vstack((NNPos, NN))
                NEdgs = np.vstack((NEdgs, np.array([NodesMat[i,0], NodeID])))

                UsedNodes.extend(list(np.where(PD[0].flatten() == NodesMat[i,0])[0]))
                WeiVal.extend(list(Wei))


        if(Mode == "QuantDists"):

            if(sum(Dists>0)==0):
                continue


            if(sum(Dists>0)>1 and len(tData)>1):
                tData_Filtered = tData[Dists>0,:]

                def DistFun(NodePosition):
                    return(np.quantile(project_point_onto_edge(X = tData_Filtered,
                                             NodePositions = np.vstack((
                                                 TargetPG['NodePositions'][NodesMat[i,0],],
                                                 NodePosition)),
                                             Edge = [0,1])['Distance_Squared'],ControlPar))



                StartingPoint = tData_Filtered[np.argmin(np.array([DistFun(tData_Filtered[i]) for i in range(len(tData_Filtered))])),:]

                if(DoSA):

                    print("Performing simulated annealing. This may take a while")
                    StartingPoint = scipy.optimize.dual_annealing(DistFun, 
                                                                  bounds=list(zip(np.min(tData_Filtered,axis=0),np.max(tData_Filtered,axis=0))),
                                                                  x0 = StartingPoint,maxiter=DoSA_maxiter)['x']


                Projections = project_point_onto_edge(X = tData_Filtered,
                                                    NodePositions = np.vstack((
                                                        TargetPG['NodePositions'][NodesMat[i,0],:],
                                                        StartingPoint)),
                                                    Edge = [0,1], ExtProj = True)

                SelId = np.argmax(
                    PartialDistance(Projections['X_Projected'],
                                              np.array([TargetPG['NodePositions'][NodesMat[i,0],:]]))
                )

                StartingPoint = Projections['X_Projected'][SelId,:]


            else:
                StartingPoint = tData[Dists>0,:]

            if not init:
                init=True
                NNPos = StartingPoint[None].copy()
                NEdgs = np.array([[NodesMat[i,0], NodeID]])
                UsedNodes.extend(list(np.where(PD[0].flatten() == NodesMat[i,0])[0]))
            else:
                NNPos = np.vstack((NNPos, StartingPoint[None]))
                NEdgs = np.vstack((NEdgs, np.array([[NodesMat[i,0], NodeID]])))
                UsedNodes.extend(list(np.where(PD[0].flatten() == NodesMat[i,0])[0]))

    # plot(X)
    # points(TargetPG$NodePositions, col="red")
    # points(NNPos, col="blue")
    # 
    TargetPG['NodePositions'] = np.vstack((TargetPG['NodePositions'], NNPos))
    TargetPG['Edges'] = [np.vstack((TargetPG['Edges'][0], NEdgs)), #edges
                         np.append(TargetPG['Edges'][1], np.repeat(np.nan, len(NEdgs)))] #lambdas
                         #np.append(TargetPG['Edges'][2], np.repeat(np.nan, len(NEdgs)))] ##### mus become lambda in R, bug ??


    if(PlotSelected):

        if(Mode == "QuantCentroid"):
            Cats = ["Unused"] * len(X)
            if(UsedNodes):
                Cats[UsedNodes] = "Used"


            p = PlotPG(X = X, TargetPG = TargetPG, GroupsLab = Cats)
            print(p)


        if(Mode == "WeightedCentroid"):
            Cats = np.zeros(len(X))
            if(UsedNodes):
                Cats[UsedNodes] = WeiVal

            p = PlotPG(X = X[Cats>0,:], TargetPG = TargetPG, GroupsLab = Cats[Cats>0])
            print(p)

            p1 = PlotPG(X = X, TargetPG = TargetPG, GroupsLab = Cats)
            print(p1)

    return(TargetPG)

def pseudo_time(root_node,point_index,traj,projval,edgeid,edges):
    """
      Quantification of pseudotime value for a single data point point_index
    """
    xi = int(point_index)
    proj_val_x = projval[xi]
    #print(proj_val_x)
    if proj_val_x<0:
        proj_val_x=0
    if proj_val_x>1:
        proj_val_x=1
    edgeid_x = edgeid[xi]
    #print(edges[edgeid_x])
    traja = np.array(traj)
    i1 = 1000000
    i2 = 1000000
    if (edges[edgeid_x][0] in traja) & (edges[edgeid_x][1] in traja):
        i1 = np.where(traja==edges[edgeid_x][0])[0][0]
        i2 = np.where(traja==edges[edgeid_x][1])[0][0]
    i = min(i1,i2)
    if i==i1:
        pstime = i1+proj_val_x
    else:
        pstime = i1-proj_val_x
    if (i1>100) | (i2>100):
         print('Edgeid_x',edgeid_x,'[0]',edges[edgeid_x][0],'[1]',edges[edgeid_x][1])
         print('proj_val_x',proj_val_x)
         print('traja',traja,'\n')
    return pstime

def pseudo_time_trajectory(traj_vertices,traj_edges,ProjStruct):
    """
      Auxillary function for quantify_pseudotime() function, should not be used
      directly
    """
    projval = ProjStruct['ProjectionValues']
    edgeid = (ProjStruct['EdgeID']).astype(int)
    edges = ProjStruct['Edges']
    partition = ProjStruct['Partition']
    traj_points = np.zeros(0,'int32')
    for p in traj_edges:
        traj_points = np.concatenate((traj_points,np.where(edgeid==p)[0]))
    #print(len(traj_points))
    pst = np.zeros(len(traj_points))
    for i,p in enumerate(traj_points):
        pst[i] = pseudo_time(traj_vertices[0],p,traj_vertices,projval,edgeid,edges)
    return pst,traj_points

def extract_trajectories(tree,root_node,verbose=False):
    """
	Extracting trajectories from ElPiGraph result object tree,
        starting from a root_node.
        Extracting trajectories is a required step for quantifying pseudotime 
        after.
        Example:
            all_trajectories,all_trajectories_edges = extract_trajectories(tree,root_node)
            print(len(all_trajectories),' trajectories found.')
            ProjStruct = project_on_tree(X,tree)
            PseudoTimeTraj = quantify_pseudotime(all_trajectories,all_trajectories_edges,ProjStruct)
    """
    edges = tree['Edges'][0]
    nodes_positions = tree['NodePositions']
    g = igraph.Graph()
    g.add_vertices(len(nodes_positions))
    g.add_edges(edges)
    degs = g.degree()
    leaf_nodes = [i for i,d in enumerate(degs) if d==1]
    if verbose:
        print(len(leaf_nodes),'trajectories found')
    all_trajectories_vertices = []
    all_trajectories_edges = []
    for lf in leaf_nodes:
        path_vertices=g.get_shortest_paths(root_node,to=lf,output='vpath')
        all_trajectories_vertices.append(path_vertices[0])
        path_edges=g.get_shortest_paths(root_node,to=lf,output='epath')
        all_trajectories_edges.append(path_edges[0])
        if verbose:
            print('Vertices:',path_vertices)
            print('Edges:',path_edges)
        ped = []
        for ei in path_edges[0]:
            ped.append((g.get_edgelist()[ei][0],g.get_edgelist()[ei][1]))
        if verbose:
            print('Edges:',ped)
        # compute pseudotime along each path
    return all_trajectories_vertices, all_trajectories_edges

def correlation_of_variable_with_trajectories(PseudoTimeTraj,var,var_names,X_original,verbose=False,producePlot=False,Correlation_Threshold=0.5):
    """
      Computes a correlation of a variable with pseudotime along all trajectories
      stored in PseudoTimeTraj structure
    """
    List_of_Associations = []
    for i,pstt in enumerate(PseudoTimeTraj):
        inds = pstt['Trajectory']
        #traj_nodep = nodep_original[inds,:]
        points = pstt['Points']
        pst = pstt['Pseudotime']
        TrajName = 'Trajectory:'+str(pstt['Trajectory'][0])+'--'+str(pstt['Trajectory'][-1])
        k = var_names.index(var)
        vals = X_original[:,k]
        spcorr = spearmanr(pst,vals[points]).correlation
        asstup = (TrajName,var,spcorr)
        if abs(spcorr)>Correlation_Threshold:
            List_of_Associations.append(asstup)
            if verbose:
                print(i,asstup)
            if producePlot:
                x = pst
                y = vals[points]
                bincenters,wav = moving_weighted_average(x,y,step_size=1.5)
                plt.plot(pst,y,'ro')
                plt.plot(bincenters,fill_gaps_in_number_sequence(wav),'bo-',linewidth=10,markersize=10)
                #plt.plot(np.linspace(0,len(inds)-1,len(inds)),traj_nodep[:,k])
                #plt.ylim(min(y)-np.ptp(y)*0.05,max(y)+np.ptp(y)*0.05)
                plt.xlabel('Pseudotime',fontsize=20)
                plt.ylabel(var,fontsize=20)
                plt.title(TrajName+', r={:2.2f}'.format(spcorr),fontsize=20)
                plt.show()
    return List_of_Associations


def regress_variable_on_pseudotime(pseudotime,vals,TrajName,var_name,var_type,producePlot=True,verbose=False,Continuous_Regression_Type='linear',R2_Threshold=0.5,max_sample=-1,alpha_factor=2):
    """
      Auxillary function for regression_of_variable_with_trajectories()

      Continuous_Regression_Type can be 'linear','gpr' for Gaussian Process, 'kr' for kernel ridge

    """
    if var_type=='BINARY':
        #convert back to binary vals
        mn = min(vals)
        mx = max(vals)
        vals[np.where(vals==mn)] = 0
        vals[np.where(vals==mx)] = 1
        if len(np.unique(vals))==1:
            regressor = None
        else:
            regressor = LogisticRegression(random_state=0,max_iter=1000,penalty='none').fit(pseudotime, vals)
    if var_type=='CATEGORICAL':
        if len(np.unique(vals))==1:
            regressor = None
        else:
            regressor = LogisticRegression(random_state=0,max_iter=1000,penalty='none').fit(pseudotime, vals)
    if var_type=='CONTINUOUS' or var_type=='ORDINAL':
        if len(np.unique(vals))==1:
            regressor = None
        else:
            if Continuous_Regression_Type=='gpr':
                # subsampling if needed
                pst = pseudotime.copy()
                vls = vals.copy()
                if max_sample>0:
                    l = list(range(len(vals)))
                    random.shuffle(l)
                    index_value = random.sample(l, min(max_sample,len(vls)))
                    pst = pst[index_value]
                    vls = vls[index_value]
                if len(np.unique(vls))>1:
                     gp_kernel =  C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
                     #gp_kernel =  RBF(np.std(vals))
                     regressor = GaussianProcessRegressor(kernel=gp_kernel,alpha=np.var(vls)*alpha_factor)
                     regressor.fit(pst, vls)
                else:
                     regressor = None
            if Continuous_Regression_Type=='linear':
                regressor = LinearRegression()
                regressor.fit(pseudotime, vals)
    
    r2score = 0
    if regressor is not None:
        r2score = r2_score(vals,regressor.predict(pseudotime))
    
        if producePlot and r2score>R2_Threshold:
            plt.plot(pseudotime,vals,'ro',label='data')
            unif_pst = np.linspace(min(pseudotime),max(pseudotime),100)
            pred = regressor.predict(unif_pst)
            if var_type=='BINARY' or var_type=='CATEGORICAL':
                prob = regressor.predict_proba(unif_pst)
                plt.plot(unif_pst,prob[:,1],'g-',linewidth=2,label='proba')    
            if var_type=='CONTINUOUS' or var_type=='ORDINAL':
                plt.plot(unif_pst,pred,'g-',linewidth=2,label='predicted')
            bincenters,wav = moving_weighted_average(pseudotime,vals.reshape(-1,1),step_size=1.5)
            plt.plot(bincenters,fill_gaps_in_number_sequence(wav),'b-',linewidth=2,label='sliding av')
            plt.xlabel('Pseudotime',fontsize=20)
            plt.ylabel(var_name,fontsize=20)
            plt.title(TrajName+', r2={:2.2f}'.format(r2score),fontsize=20)
            plt.legend(fontsize=15)
            plt.show()

    
    return r2score, regressor

def regression_of_variable_with_trajectories(PseudoTimeTraj,var,var_names,variable_types,X_original,verbose=False,producePlot=True,R2_Threshold=0.5,Continuous_Regression_Type='linear',max_sample=1000,alpha_factor=2):
    """
      Regression analysis of a single variable against all trajectories
      stored in PseudoTimeTraj

      Continuous_Regression_Type can be 'linear','gpr' for Gaussian Process, 'kr' for kernel ridge

    """
    List_of_Associations = []
    for i,pstt in enumerate(PseudoTimeTraj):
        inds = pstt['Trajectory']
        #traj_nodep = nodep_original[inds,:]
        points = pstt['Points']
        pst = pstt['Pseudotime']
        pst = pst.reshape(-1,1)
        TrajName = 'Trajectory:'+str(pstt['Trajectory'][0])+'--'+str(pstt['Trajectory'][-1])
        k = var_names.index(var)
        vals = X_original[points,k]
        r2,regressor = regress_variable_on_pseudotime(pst,vals,TrajName,var,variable_types[k],producePlot=producePlot,verbose=verbose,R2_Threshold=R2_Threshold,Continuous_Regression_Type=Continuous_Regression_Type, max_sample=max_sample,alpha_factor=alpha_factor)
        pstt[var+'_regressor'] = regressor
        asstup = (TrajName,var,r2)
        #if verbose:
        #    print(var,'R2',r2)
        if r2>R2_Threshold:
            List_of_Associations.append(asstup)
            if verbose:
                print(i,asstup)
    return List_of_Associations


def quantify_pseudotime(all_trajectories_vertices,all_trajectories_edges,ProjStruct,producePlot=False):
    """
      Main function for quantifying pseudotime

        Example of use:
            all_trajectories,all_trajectories_edges = extract_trajectories(tree,root_node)
            print(len(all_trajectories),' trajectories found.')
            ProjStruct = project_on_tree(X,tree)
            PseudoTimeTraj = quantify_pseudotime(all_trajectories,all_trajectories_edges,ProjStruct)

    """
    projval = ProjStruct['ProjectionValues']
    edgeid = (ProjStruct['EdgeID']).astype(int)
    edges = ProjStruct['Edges']
    partition = ProjStruct['Partition']
    PseudoTimeTraj = []
    for i,traj in enumerate(all_trajectories_vertices):
        pst,points = pseudo_time_trajectory(traj,all_trajectories_edges[i],ProjStruct)
        pstt = {}
        pstt['Trajectory'] = traj
        pstt['Trajectory_Edges'] = all_trajectories_edges[i]
        pstt['Points'] = points
        pstt['Pseudotime'] = pst
        PseudoTimeTraj.append(pstt)
        if producePlot:
            plt.plot(np.sort(pst))
    return PseudoTimeTraj

def project_on_tree(X,tree):
    """
       Project a dataset X on ElPiGraph result structure tree.

       Returns ProjStruct structure
    """
    nodep = tree['NodePositions']
    edges = tree['Edges'][0]
    partition, dists = elpigraph.src.core.PartitionData(X = X, NodePositions = nodep, MaxBlockSize = 100000000, TrimmingRadius = np.inf,SquaredX = np.sum(X**2,axis=1,keepdims=1))
    ProjStruct = elpigraph.src.reporting.project_point_onto_graph(X = X,
                                     NodePositions = nodep,
                                     Edges = edges,
                                     Partition = partition)
    #projval = ProjStruct['ProjectionValues']
    #edgeid = (ProjStruct['EdgeID']).astype(int)
    ProjStruct['Partition'] = partition
    return ProjStruct

def draw_pseudotime_dependence(trajectory,variable_name,variable_names,variable_types,X_original,color_line,linewidth=1,fontsize=20,draw_datapoints=False,label=None,linestyle=None):
    """
       Drawing function showing dependence of a variable on pseudotime along
       a particular trajectory
    """
    regressor = trajectory[variable_name+'_regressor']
    k = variable_names.index(variable_name)
    mn = min(X_original[:,k])
    mx = max(X_original[:,k])
    vals = None
    if regressor is not None:
        pst = trajectory['Pseudotime']
        #pst = np.unique(pst).reshape(-1,1)
        unif_pst = np.linspace(min(pst),max(pst),100).reshape(-1,1)
        var_type = variable_types[k]
        if var_type=='BINARY':
            vals = regressor.predict_proba(unif_pst)[:,1]
        else:
            vals = regressor.predict(unif_pst)
            vals = (vals-mn)/(mx-mn)
        if draw_datapoints:
            plt.plot(pst,(X_original[trajectory['Points'],k]-mn)/(mx-mn),'ko',color=color_line)
        if label is None:
            label = variable_name
        if linestyle is None:
            linestyle = '-'
        plt.plot(unif_pst,vals,color=color_line,linewidth=linewidth,label=label,linestyle=linestyle)
        plt.xlabel('Pseudotime',fontsize=fontsize)
    return vals

def add_pie_charts_tree(ax,tree,values,color_seq,scale=1):
    """
       Drawing function used to draw pie-charts on top of a principal tree layout
    """
    nodep = tree['NodePositions']
    edges = tree['Edges'][0]
    g=nx.Graph()
    g.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(g,scale=2)
    idx=np.array([pos[j] for j in range(len(pos))])
    add_pie_charts(ax,idx,values,color_seq,scale=scale)

def add_pie_charts(ax,node_positions2d,values,color_seq,partition,scale=1):
    """
       Auxillary function for add_pie_charts_tree()
    """
    df = pd.DataFrame({'CLASS':values})
    vals_unique_df = df.CLASS.value_counts()
    vals_unique = vals_unique_df.index.to_list()
    vals_unique_freq = vals_unique_df.to_numpy()
    print(vals_unique,vals_unique_freq)

    for i in range(node_positions2d.shape[0]):
        inode = np.where(partition==i)
        dfi = df.loc[partition==i]
        node_valunique_df = dfi.CLASS.value_counts()
        node_valunique = node_valunique_df.index.to_list()
        node_valunique_freq = node_valunique_df.to_numpy()
        freq_sum = np.sum(node_valunique_freq)
        freq = len(vals_unique)*[0]
        for j,v in enumerate(node_valunique):
            freq[vals_unique.index(v)] = node_valunique_freq[j]/freq_sum
        #print(i,':',node_valunique,node_valunique_freq)
        #print(i,':',freq)
        draw_pie(ax,freq,color_seq,X=node_positions2d[i,0],Y=node_positions2d[i,1],size=scale*len(inode[0]))

def draw_pie(ax,ratios,colors,X=0, Y=0, size = 1000):
    """
       Auxillary function for add_pie_charts_tree()
    """
    N = len(ratios)
    xy = []
    start = 0.
    for ratio in ratios:
        x = [0] + np.cos(np.linspace(2*math.pi*start,2*math.pi*(start+ratio), 30)).tolist()
        y = [0] + np.sin(np.linspace(2*math.pi*start,2*math.pi*(start+ratio), 30)).tolist()
        xy1 = list(zip(x,y))
        xy.append(xy1)
        start += ratio
    for i, xyi in enumerate(xy):
        ax.scatter([X],[Y] , marker=xyi, s=size, facecolor=colors[i] )
