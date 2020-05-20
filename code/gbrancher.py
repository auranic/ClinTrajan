
'''
vec_labels_by_branches = branch_labler( X , g, nodes_positions )
dict_output = find_branches(g, verbose = 1000)
'''

import numpy as np

def partition_data(Xcp, NodePositions, MaxBlockSize = 10**6,SquaredXcp= None,
                  TrimmingRadius=float('inf')):
    '''
    # Partition the data by proximity to graph nodes
    # (same step as in K-means EM procedure)
    #
    # Inputs:
    #   X is n-by-m matrix of datapoints with one data point per row. n is
    #       number of data points and m is dimension of data space.
    #   NodePositions is k-by-m matrix of embedded coordinates of graph nodes,
    #       where k is number of nodes and m is dimension of data space.
    #   MaxBlockSize integer number which defines maximal number of
    #       simultaneously calculated distances. Maximal size of created matrix
    #       is MaxBlockSize-by-k, where k is number of nodes.
    #   SquaredX is n-by-1 vector of data vectors length: SquaredX = sum(X.^2,2);
    #   TrimmingRadius (optional) is squared trimming radius.
    #
    # Outputs
    #   partition is n-by-1 vector. partition[i] is number of the node which is
    #       associated with data point X[i, ].
    #   dists is n-by-1 vector. dists[i] is squared distance between the node with
    #       number partition[i] and data point X[i, ].
    '''
    if SquaredXcp is None:
      SquaredXcp = np.sum(Xcp**2,1)[:,np.newaxis]
    NodePositionscp = np.asarray(NodePositions)
    n = Xcp.shape[0]
    partition = np.zeros((n, 1), dtype=int)
    dists = np.zeros((n, 1))
    all_dists = np.zeros((n, NodePositions.shape[0] ))
    # Calculate squared length of centroids
    cent = NodePositionscp.T
    centrLength = (cent**2).sum(axis=0)
    # Process partitioning without trimming
    for i in range(0, n, MaxBlockSize):
        # Define last element for calculation
        last = i+MaxBlockSize
        if last > n:
            last = n
        # Calculate distances
        d = SquaredXcp[i:last] + centrLength-2*np.dot(Xcp[i:last, ], cent)
        tmp = d.argmin(axis=1)
        partition[i:last] = tmp[:, np.newaxis]
        dists[i:last] = d[np.arange(d.shape[0]), tmp][:, np.newaxis]
        all_dists[i:last,:] = d
    # Apply trimming
    if not np.isinf(TrimmingRadius):
        ind = dists > (TrimmingRadius**2)
        partition[ind] = -1
        dists[ind] = TrimmingRadius**2
    
    
    return np.asarray(partition), np.asarray(dists), np.asarray(all_dists)



def find_branches( graph, verbose = 0 ):
  '''
  #' Computes "branches" of the graph, i.e. paths from branch vertex (or terminal vertex)  to branch vertex (or terminal vertex)
  #' Can process disconnected graphs. Stand-alone point - is "branch".
  #' Circle is exceptional case - each circle (can be several connected components) is "branch"
  #'
  #' @param g - graph (igraph) 
  #' @param verbose - details output
  #' 
  #' @examples
  #' import igraph
  #' g = igraph.Graph.Lattice([3,3], circular = False ) 
  #' dict_output = find_branches(g, verbose = 1000)
  #' print( dict_output['branches'] )
  '''
  #verbose = np.inf
  #
  g = graph
  n_vertices_input_graph =   g.vcount()  
  set_vertices_input_graph = range( n_vertices_input_graph  )

  dict_output = {}
  #dict_output['branches'] = found_branches.copy()

  # Main variables for process: 
  found_branches = []
  processed_edges = []
  processed_vertices = set()

  ############################################################################################################################################
  # Connected components loop:
  count_connected_components = 0 
  while True: # Need loop if graph has several connected components, each iteration - new component
    count_connected_components += 1

    def find_start_vertex(g, processed_vertices ): 
      '''
      #' Find starting vertex for branches-search algorithm. 
      #' It should be either branching vertex (i.e. degree >2) or terminal vertex (i.e. degree 0 or 1), in special case when unprocessed part of graph is union of circles - processed outside function
      '''
      n_vertices = n_vertices_input_graph #  = g.count()# 
      if n_vertices == len( processed_vertices ):
        return -1,-1 # All vertices proccessed
      flag_found_start_vertex = 0 
      for v in set_vertices_input_graph: 
        if v in processed_vertices: continue
        if g.degree(v) != 2:
          flag_found_start_vertex = 1
          return v, flag_found_start_vertex
      return -1, 0 # All unprocessed vertices are of degree 2, that means graph is circle of collection or collection of circles

    ############################################################################################################################################
    # Starting point initialization. End process condtion.
    #
    # Set correctly the starting vertex for the algorithm
    # That should be branch vertex or terminal vertex, only in case graph is set of circles(disconnected) we take arbitrary vertex as initial, each circle will be a branch
    initial_vertex, flag_found_start_vertex = find_start_vertex(g, processed_vertices )
    if   flag_found_start_vertex > 0:
      current_vertex  = initial_vertex
    elif flag_found_start_vertex == 0: # All unprocessed vertices are of degree 2, that means graph is circle of collection or collection of circles
      # Take any unprocessed element 
      tmp_set = set_vertices_input_graph  - processed_vertices
      current_vertex = tmp_set.pop()
    else:
      # No vertices to process 
      if verbose >= 10:
        print('Process finished')
      dict_output['branches'] = found_branches.copy()
      return dict_output
      #break

    ############################################################################################################################################
    # Core function implementing "Breath First Search" like algorithm
    # with some updates in storage, since we need to arrange edges into "branches"
    def find_branches_core( current_vertex , previous_vertex, current_branch  ):
      core_call_count[0] = core_call_count[0] + 1
      if verbose >= 1000:
        print(core_call_count[0], 'core call.', 'current_vertex', current_vertex , 'previous_vertex', previous_vertex,'found_branches',found_branches, 'current_branch',current_branch )

      processed_vertices.add(current_vertex)
      neis = g.neighbors(current_vertex) 
      if len(neis) == 0: # current_vertex is standalone vertex
        found_branches.append( [current_vertex] )
        return 
      if len(neis) == 1: # current_vertex is terminal vertex
        if neis[0] == previous_vertex:
          current_branch.append( current_vertex  )
          found_branches.append( current_branch.copy() )
          # processed_edges.append(  set([current_vertex , previous_vertex])  )  
          return 
        else:
          # That case may happen if we just started from that vertex
          # Because it has one neigbour, but it is not previous_vertex, so it is None, which is only at start 
          current_branch = [current_vertex] # , neis[0] ] # .append( current_vertex  )
          processed_edges.append(  set([current_vertex , neis[0] ])  )
          find_branches_core( current_vertex = neis[0] , previous_vertex = current_vertex, current_branch = current_branch )  
          return
      if len(neis) == 2: # 
        # continue the current branch:
        current_branch.append( current_vertex  )
        next_vertex = neis[0]
        if next_vertex == previous_vertex: next_vertex = neis[1]
        if next_vertex in processed_vertices: # Cannot happen for trees, but may happen if graph has a loop
          if set([current_vertex , next_vertex]) not in processed_edges:
            current_branch.append( next_vertex  )
            found_branches.append( current_branch.copy() )
            processed_edges.append(  set([current_vertex , next_vertex])  )
            return 
          else:
            return
        processed_edges.append(  set([current_vertex , next_vertex])  )          
        find_branches_core( current_vertex=next_vertex , previous_vertex = current_vertex, current_branch = current_branch )
        return
      if len(neis)  > 2 : #Branch point
        if  previous_vertex is not None:
          # Stop current branch
          current_branch.append( current_vertex  )
          found_branches.append(current_branch.copy())
        for next_vertex in neis:
            if next_vertex ==  previous_vertex:    continue
            if next_vertex in  processed_vertices: # Cannot happen for trees, but may happen if graph has a loop
              if set([current_vertex , next_vertex]) not in processed_edges:
                processed_edges.append(  set([current_vertex , next_vertex])  )
                found_branches.append( [current_vertex, next_vertex ] )
              continue
            current_branch = [current_vertex]
            processed_edges.append(  set([current_vertex , next_vertex])  )
            find_branches_core( current_vertex = next_vertex , previous_vertex = current_vertex , current_branch = current_branch)
      return

    ############################################################################################################################################
    # Core function call. It should process the whole connected component
    if verbose >= 10:
      print('Start process count_connected_components', count_connected_components, 'initial_vertex', current_vertex)
    processed_vertices.add(current_vertex)
    core_call_count = [0]
    find_branches_core( current_vertex = current_vertex , previous_vertex = None , current_branch = [])

    ############################################################################################################################################
    # Output of results for connected component
    if verbose >=10:
      print('Connected component ', count_connected_components, ' processed ')
      print('Final found_branches',found_branches)
      print('N Final found_branches', len( found_branches) )


def branch_labler( X , graph, nodes_positions, verbose = 0 ):
  '''
  #' Labels points of the dataset X by "nearest"-"branches" of graph.
  #' 
  #'
  #' @examples
  # X = np.array( [[0.1,0.1], [0.1,0.2], [1,2],[3,4],[5,0]] )
  # nodes_positions = np.array( [ [0,0], [1,0], [0,1], [1,1] ]  ) 
  # import igraph
  # g = igraph.Graph(); g.add_vertices(  4  )
  # g.add_edges([[0,1],[0,2],[0,3]])
  # vec_labels_by_branches = branch_labler( X , g, nodes_positions )
  '''
  #####################################################################################
  # Calculate branches and clustering by vertices of graph 
  dict_output = find_branches(graph, verbose = verbose )
  if verbose >=100:
    print('Function find_branches results branches:',  dict_output['branches'] )
  vec_labels_by_vertices, dists, all_dists = partition_data(X, nodes_positions) # np.array([[1,2,3,4], [1,2,3,4], [1,2,3,4], [10,20,30,40]]), [[1,2,3,4], [10,20,30,40]], 10**6)#,SquaredX)
  vec_labels_by_vertices = vec_labels_by_vertices.ravel()
  if verbose >=100:
    print('Function partition_data returns: vec_labels_by_vertices.shape, dists.shape, all_dists.shape', vec_labels_by_vertices.shape, dists.shape, all_dists.shape )
  #####################################################################################

  n_vertices = len( nodes_positions)
  branches = dict_output['branches']

  #####################################################################################
  # Create dictionary vertex to list of branches it belongs to  
  dict_vertex2branches = {}
  for i,b in enumerate( branches):
    for v in b:
      if v in dict_vertex2branches.keys():
        dict_vertex2branches[v].append(i)
      else:
        dict_vertex2branches[v] = [i]
  if verbose >=100:
    print( 'dict_vertex2branches', dict_vertex2branches )


  #####################################################################################
  # create list of branch and non-branch vertices
  list_branch_vertices = []
  list_non_branch_vertices = []
  for v in dict_vertex2branches:
    list_branches = dict_vertex2branches[v]
    if len(list_branches) == 1:
      list_non_branch_vertices.append(v)
    else:
      list_branch_vertices.append(v)
  if verbose >=100:  
    print('list_branch_vertices, list_non_branch_vertices', list_branch_vertices, list_non_branch_vertices)

  #####################################################################################
  # First stage of creation of final output - create labels by branches vector 
  # After that step it will be only correct for non-branch points 
  vec_vertex2branch = np.zeros(  n_vertices  ) 
  for i in range( n_vertices  ):
    vec_vertex2branch[i] = dict_vertex2branches[i][0]
  vec_labels_by_branches = vec_vertex2branch[ vec_labels_by_vertices ] 
  if verbose >= 100:
    print('branches', branches)
    print('vec_labels_by_branches', vec_labels_by_branches)

  #####################################################################################
  # Second stage of creation of final output - 
  # make correct calculation for branch-vertices create labels by correct branches 
  for branch_vertex in list_branch_vertices:
    if verbose >= 100:
      print('all_dists.shape', all_dists.shape)
    def labels_for_one_branch_vertex( branch_vertex , vec_labels_by_vertices,  all_dists ):
      '''
      #' For the branch_vertex re-labels points of dataset which were labeled by it to label by "correct branch".
      #' "Correct branch" label is a branch 'censored'-nearest to given point. 
      #' Where 'censored'-nearest means the minimal distance between the point  and all points of the branch except the given branch_vertex
      #'
      #' Function changes vec_labels_by_branches defined above
      #' Uses vec_labels_by_vertices defined above - vector of same length as dataset, which contains labels by vertices 
      '''

      mask = vec_labels_by_vertices.ravel() == branch_vertex # Select part of the dataset which is closest to branch_vertex

      # Allocate memory for array: first coordinate - point of dataset[mask],  second coordinate - branch number , for all branches contianing given vertex (i.e. branch_vertex) 
      # For each point of dataset[mask] it contains 'censored'-distances to "branches" adjoint to "branch_vertex", 
      # 'censored' means minimal over vertices belonging to  distance to branches (excluding branch_vertex)
      dist2branches = np.zeros( [ mask.sum(), len(dict_vertex2branches[branch_vertex] )  ] )
      list_branch_ids = [] # that will be necessary to renumerate local number to branch_ids 
      for i,branch_id in enumerate( dict_vertex2branches[branch_vertex] ):
        list_branch_ids.append(branch_id)
        # Create list of vertices of current branch, with EXCLUSION of branch_vertex
        branch_vertices_wo_given_branch_vertex = [v for v in branches[branch_id] if v != branch_vertex ]
        # For all points of dataset[mask] calculate minimal distances to given branch (with exclusion of branch_point), i.e. mininal difference for  
        if verbose >= 1000:
          print('mask.shape, all_dists.shape', mask.shape, all_dists.shape)
        dist2branches[ : ,i ] = np.min( all_dists[mask,:][:,branch_vertices_wo_given_branch_vertex], 1 ).ravel()

      vec_labels_by_branches[mask] = np.array(list_branch_ids)[ np.argmin( dist2branches, 1) ]
    labels_for_one_branch_vertex( branch_vertex, vec_labels_by_vertices,  all_dists  )

    if verbose >= 10:    
      print('Output: vec_labels_by_branches', vec_labels_by_branches)


  return vec_labels_by_branches
