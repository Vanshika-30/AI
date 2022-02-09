'''
    Name: Vanshika Jain
    Roll No.: BT18CSE107
    Algorithm: A-start on TSP 
    Heuristic: Minimum Spanning Tree (Prims)
'''
import sys
import heapq
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import networkx as nx
   
# Defining state class to store the current arrangement of states
class State:
    def __init__(self, V, curr_node, parent, h, visited, g):
        # storing id of the state, curr_node = node number 
        self.curr_node = curr_node
        # store parent of this state
        self.parent = parent
        self.h = h
        self.g = g
        self.f = self.g + self.h
        # to keep track of path and nodes visited
        self.visited = visited

    def __lt__(self, other):
        return self.f < other.f

# Function to add edge to a graph
def adding_new_edge_to_graph(graph, node1, node2, cost):
  
    graph[node1][node2] = cost
    graph[node2][node1] = cost

# Util function for finding MST
def minKey(min_weight_edge, mst_visited, nodes):
    # Initialize min value
    # if no index exists then MST not possible and return h = inf 
    min_index = -1
    min = sys.maxsize
    for v in nodes:
        if min_weight_edge[v] < min and mst_visited[v] == False:
            min = min_weight_edge[v]
            min_index = v
    return min_index

# Function to find MST
def find_MST(graph, nodes):
    nodes = list(nodes)
    # if no unvisited nodes left
    if len(nodes) == 0:
        return 0
    
    # no of vertices
    V = len(graph)
    # to pick minimum edge weight from the remaining graph
    min_weight_edge = [sys.maxsize] * V
    # to store MST path
    mst_path = [None] * V 
    # considering 1st state of the unvisited set
    min_weight_edge[nodes[0]] = 0
    mst_visited = [False] * V
 
    mst_path[nodes[0]] = -1 # First node is always the root of

    for cout in range(len(nodes)):
        # Pick minimum distance vertex from the set not processed yet 
        u = minKey(min_weight_edge, mst_visited, nodes)

        # in case the unvisited states dont form a connected graph
        if u == -1:
            return sys.maxsize
        
        # visited the min weight node
        mst_visited[u] = True

        # update dist value of adjacent vertices of the picked 
        # one if current dist > new_dist 
        # and vertex not yet visited

        for v in nodes:
            # checking if this edge exists in graph and updating they min_edge_weight and mst_path
            c = graph[u].get(v, None)
            if c is not None:
                if c > 0 and mst_visited[v] == False and min_weight_edge[v] > c:
                    min_weight_edge[v] = c
                    mst_path[v] = u

    # Calculating MST cost
    cost = 0
    for i in nodes:
        c = graph[i].get(mst_path[i], None)
        if c is None:
            c = 0
        cost += c

    return cost

# Finding successors and calculating the heuristic 
def find_successors(graph, curr_state):
    # stroing successors of curr_state
    successors = []
    no_generate = 0
    for s,c in graph[curr_state.curr_node].items():
        if not curr_state.visited[s]:
            # copying the visited of parent as these states have been done in this path
            visited = copy.deepcopy(curr_state.visited)

            # making the current neighbour as visited
            visited[s] = True

            # finding list of unvisited nodes
            unvisited = []
            for i,v in enumerate(visited):
                if not v:
                    unvisited.append(i)
            h_val = find_MST(graph, unvisited)
            # updating the f,g,h values in the new State
            temp_state = State(V, s, curr_state, h_val, visited, curr_state.g + c)
            successors.append(temp_state)
            no_generate+=1
    return successors, no_generate

def find_optimal_tsp_path(graph, V):
    
    nodes_generate = 0
    nodes_expanded = 0

    # MST of whole graph initially
    h_val = find_MST(graph, sorted(graph.keys()))
    curr_state = State(V, 0, None, h_val, [False]*V, 0)
    
    # heap to pick the minimum state node from all the generated nodes based on f-val
    fringe_list = []
    fringe_list.append(curr_state)
    heapq.heapify(fringe_list)
   
    # if fringe_list becomes empty, no solution exists
    while fringe_list:
        curr_state = heapq.heappop(fringe_list)
        nodes_expanded +=1

        # if all vertices visited and back to start node, soltuion found
        if all (curr_state.visited) and curr_state.curr_node == 0:
            print("\nTSP Solved at cost ", curr_state.g, '\n')
            print("Path of MSP: ", end = " ")
            path = []
            while curr_state.parent  is not None:
                path.append(curr_state.curr_node)
                # print(curr_state.curr_node, end = " ")
                curr_state = curr_state.parent
            path.append(0)
            return path, nodes_generate, nodes_expanded
        
        # find successors of current vertice and append to fringe list
        else:
            successors, gen = find_successors(graph, curr_state)
            nodes_generate += gen
            for s in successors:
                heapq.heappush(fringe_list, s)

    print("TSP not possible\n")
    return [], nodes_generate, nodes_expanded

def dfs(graph, visited, source):
    if visited[source] == True:
        return
    visited[source] = True
    for neighbours in graph[source]:
        dfs(graph, visited, neighbours)

def check_connected(graph, source):
    visited = [False]*len(graph)
    dfs(graph, visited, source)
    
    j = 0
    for i in visited:
        if i == False:
            return False
        j+=1
    return True

if __name__ == '__main__': 

    # run pip install networkx if netwrokx module not installed for visualization of final path

    graph = defaultdict(dict) #store data values like a map key:value pair
    graph_input = None
    path = []

    f = open("input_25.txt","r")
    lines = f.readline()
    V = int(lines)

    i = 0
    # input file is in the form of adjacency matrix, so number of lines = total no of nodes in the graph

    lines = f.readlines()
    count = 0  # counts number of nodes
    for line in lines:
        count+=1
        graph_input = line.split(" ")
        graph_input = graph_input[0:len(graph_input)-1]
        if graph_input is not None:
            j = 0
            for c in graph_input:
                if c != '-1':
                    adding_new_edge_to_graph(graph,i,j,float(c))
                j+=1
        i+=1

    if len(graph) != V:
        i = len(graph)-1

        while i < count:
            adding_new_edge_to_graph(graph,i,i,0)
            i+=1

    # print(graph)
    if len(graph) != 0:
        # Call Tsp path function which uses A* algorithm to exapnd nodes in order of non decresing f-values using
        # MST as admisible Heurestics (Prim's algorithm for MST)
        # print(find_optimal_tsp_path(graph))
        if check_connected(graph, list(graph.keys())[0]):
            
            path, generated_nodes, expanded_nodes = find_optimal_tsp_path(graph, V) 
            print(path, '\n')
            print("Number of Expanded Nodes : ", expanded_nodes, '\n')
            print("Number of Generated nodes in fringe list : ", generated_nodes, '\n')
        else:
            print("\nTSP Not possible as graph not connected\n")

    else:
        print("\nGiven Graph is empty\n")

    # Displaying original graph
    G1=nx.Graph()
    node_label = {}
    for i in range(V):
        G1.add_node(i)
        node_label[i] = str(i)

    for n1 in graph:
        for n2,c in graph[n1].items():
            G1.add_edge(n1, n2, weight=c)
    pos=nx.circular_layout(G1)
    nx.draw(G1,pos)
    labels = nx.get_edge_attributes(G1,'weight')
    nx.draw_networkx_labels(G1,pos,node_label,font_size=16)
    nx.draw_networkx_edge_labels(G1,pos, edge_labels=labels)

    # displaying TSP Path
    if len(path) > 0:  
        G=nx.Graph()
        node_label = {}
        for i in range(V):
            G.add_node(i)
            node_label[i] = str(i)

        for i in range(1,len(path)):
            G.add_edge(path[i-1], path[i],color='red', weight=graph[path[i-1]][path[i]])
        pos=nx.circular_layout(G)
        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]
        nx.draw(G,pos, edge_color=colors)
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_labels(G,pos,node_label,font_size=16)
        nx.draw_networkx_edge_labels(G,pos, edge_labels=labels)
    plt.show()