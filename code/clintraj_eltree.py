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

import scipy.optimize
import copy
import warnings
from elpigraph.src.graphs import ConstructGraph, GetSubGraph, GetBranches
from elpigraph.src.core import PartitionData
from elpigraph.src.distutils import PartialDistance
from elpigraph.src.reporting import project_point_onto_graph, project_point_onto_edge


def visualize_eltree_with_data(tree_elpi,X,X_original,principal_component_vectors,mean_vector,color,variable_names,
                              showEdgeNumbers=False,showNodeNumbers=False,showBranchNumbers=False,showPointNumbers=False,
                              Color_by_feature = '', Feature_Edge_Width = '', Invert_Edge_Value = False,
                              Min_Edge_Width = 10, Max_Edge_Width = 10, 
                              Big_Point_Size = 100, Small_Point_Size = 1, Normal_Point_Size = 30,
                              Visualize_Edge_Width_AsNodeCoordinates=False,
                              Color_by_partitioning = False,
                              vec_labels_by_branches = [],
                              Transparency_Alpha = 0.2,
                              Visualize_Branch_Class_Associations = [], #list_of_branch_class_associations
                              cmap = 'cool',scatter_parameter=0.03,highlight_subset=[]):

    nodep = tree_elpi['NodePositions']
    nodep_original = np.matmul(nodep,principal_component_vectors[:,0:X.shape[1]].T)+mean_vector
    adjmat = tree_elpi['ElasticMatrix']
    edges = tree_elpi['Edges'][0]
    color2 = color
    if not Color_by_feature=='':
        k = variable_names.index(Color_by_feature)
        color2 = X_original[:,k]
    if Color_by_partitioning:
        color2 = vec_labels_by_branches
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
            k1 = k%len(color_seq)
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
    print('Partitioning the data...')
    partition, dists = elpigraph.src.core.PartitionData(X = X, NodePositions = nodep, MaxBlockSize = 100000000, TrimmingRadius = np.inf,SquaredX = np.sum(X**2,axis=1,keepdims=1))
    #col_nodes = {node: color[np.where(partition==node)[0]] for node in np.unique(partition)}


    #Project points onto the graph
    print('Projecting data points onto the graph...')
    ProjStruct = elpigraph.src.reporting.project_point_onto_graph(X = X,
                                     NodePositions = nodep,
                                     Edges = edges,
                                     Partition = partition)

    projval = ProjStruct['ProjectionValues']
    edgeid = (ProjStruct['EdgeID']).astype(int)
    X_proj = ProjStruct['X_projected']

    dist2proj = np.sum(np.square(X-X_proj),axis=1)
    shift = np.percentile(dist2proj,20)
    dist2proj = dist2proj-shift

    #Create graph
    print('Producing graph layout...')
    g=nx.Graph()
    g.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(g,scale=2)
    #pos = nx.planar_layout(g)
    #pos = nx.spring_layout(g,scale=2)
    idx=np.array([pos[j] for j in range(len(pos))])

    plt.figure(figsize=(16,16))

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
    
        #random angle
        #alpha = 2 * np.pi * np.random.random()
        #random scatter to appropriate distance 
        #x[i] = r * np.cos(alpha) + x_coo
        #y[i] = r * np.sin(alpha) + y_coo
	# we rather position the point close to project and put
	# it at distance r orthogonally to the edge 
	# on a random side of the edge 
        vex = x_coos[1]-x_coos[0]
        vey = y_coos[1]-y_coos[0]
        vn = np.sqrt(vex*vex+vey*vey)
        vex = vex/vn
        vey = vey/vn
        rsgn = random_sign()
        x[i] = x_coo+vey*r*rsgn
        y[i] = y_coo-vex*r*rsgn

    plt.scatter(x,y,c=color2,cmap=cmap,s=points_size, vmin=min(color2), vmax=max(color2))
    if showPointNumbers:
        for j in range(len(X)):
            plt.text(x[j],y[j],j)
    if len(highlight_subset)>0:
        [color_subset] = [color2[i] for i in highlight_subset]
        plt.scatter(x[highlight_subset],y[highlight_subset],c=color_subset,cmap=cmap,s=Big_Point_Size)
    plt.colorbar()


    #Scatter nodes
    plt.scatter(idx[:,0],idx[:,1],s=node_size,c='black',alpha=.8)

    #Associate edge width to a feature
    edge_vals = [1]*len(edges)
    if not Feature_Edge_Width=='' and not Visualize_Edge_Width_AsNodeCoordinates:
        k = variable_names.index(Feature_Edge_Width)
        for j in range(len(edges)):
            vals = X_original[np.where(edgeid==j),k]
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
        branch_vals = list(set(vec_labels_by_branches))
        for i,val in enumerate(branch_vals):
            ind = vec_labels_by_branches==val
            xbm = np.mean(x[ind])
            ybm = np.mean(y[ind])
            plt.text(xbm,ybm,int(val),FontSize=20,bbox=dict(facecolor='grey', alpha=0.5))
        
    if showNodeNumbers:
        for i in range(nodep.shape[0]):
            plt.text(idx[i,0],idx[i,1],str(i),FontSize=20,bbox=dict(facecolor='grey', alpha=0.5))

    #plt.axis('off')


def partition_data_by_tree_branches(X,tree_elpi):
    edges = tree_elpi['Edges'][0]
    nodes_positions = tree_elpi['NodePositions']
    g = igraph.Graph()
    g.add_vertices(len(nodes_positions))
    g.add_edges(edges)
    vec_labels_by_branches = branch_labler( X , g, nodes_positions )
    return vec_labels_by_branches


def prune_the_tree(tree_elpi):
    # remove short 'orphan' edges not forming a continuous branch
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
    if edges[edgeid_x][0] in traja:
        i1 = np.where(traja==edges[edgeid_x][0])[0][0]
    if edges[edgeid_x][1] in traja:
        i2 = np.where(traja==edges[edgeid_x][1])[0][0]
    i = min(i1,i2)
    pstime = i+proj_val_x
    return pstime

def pseudo_time_trajectory(traj,projval,edgeid,edges,partition):
    traj_points = np.zeros(0,'int32')
    for p in traj:
        traj_points = np.concatenate((traj_points,np.where(partition==p)[0]))
    #print(len(traj_points))
    pst = np.zeros(len(traj_points))
    for i,p in enumerate(traj_points):
        pst[i] = pseudo_time(traj[0],p,traj,projval,edgeid,edges)
    return pst,traj_points

def extract_trajectories(tree,root_node,verbose=False):
    edges = tree['Edges'][0]
    nodes_positions = tree['NodePositions']
    g = igraph.Graph()
    g.add_vertices(len(nodes_positions))
    g.add_edges(edges)
    degs = g.degree()
    leaf_nodes = [i for i,d in enumerate(degs) if d==1]
    if verbose:
        print(len(leaf_nodes),'trajectories found')
    all_trajectories = []
    for lf in leaf_nodes:
        path_vertices=g.get_shortest_paths(root_node,to=lf,output='vpath')
        all_trajectories.append(path_vertices[0])
        path_edges=g.get_shortest_paths(root_node,to=lf,output='epath')
        if verbose:
            print('Vertices:',path_vertices)
            print('Edges:',path_edges)
        ped = []
        for ei in path_edges[0]:
            ped.append((g.get_edgelist()[ei][0],g.get_edgelist()[ei][1]))
        if verbose:
            print('Edges:',ped)
        # compute pseudotime along each path
    return all_trajectories