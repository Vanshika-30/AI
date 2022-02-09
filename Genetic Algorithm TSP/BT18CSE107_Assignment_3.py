"""
    Name: Vanshika Jain
    Roll No.: BT18CSE107
    Algorithm: Genetic Algorithm on TSP 
    Crossover: Basic Crossover, Ordered Crossover, PMX Crossover
    Mutation: Swapping the nodes between 2 random indexes
    To run: python BT18CSE107_Assignment_3.py --popsize 1000 --gen 200 --inputfile inputs/input_25.txt (path to file)
"""

import heapq, argparse
from random import randint
import math
import numpy as np
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import networkx as nx

INF = math.inf
graph = defaultdict(dict)

# Function to add edge to a graph
def adding_new_edge_to_graph(node1, node2, cost):
    global graph
    graph[node1][node2] = cost
    graph[node2][node1] = cost

# Function to calculate cost of a given path
def calculate_tsp_cost(pop):
    global graph
    cost = 0
    for i in range(len(pop) - 1):
        if (pop[i]) in graph and (pop[i + 1]) in graph[pop[i]]:
            cost += graph[pop[i]][pop[i + 1]]
        else:
            cost = INF
            break
    if (pop[0]) in graph and (pop[-1]) in graph[pop[0]]:
        cost += graph[pop[0]][pop[-1]]
    else:
        cost = INF
    return cost

'''
  Function to generate population randomly
  Storing the path and 1/cost or fitness of the path in heap
'''
def generate_population(V, population_size):
    global graph
    tsp = tuple([i for i in range(V)])
    population = []
    heapq.heapify(population)
    for _ in range(population_size):
        permutation = np.random.permutation(tsp)
        cost = calculate_tsp_cost(permutation)
        if cost != INF:
            heapq.heappush(population, tuple([1/cost, tuple(permutation)]))
        else:
            heapq.heappush(population, tuple([0, tuple(permutation)]))
    return population

'''
  Mutation:
    Selecting 2 index k and l and swapping elements between those 2 to get an offspring
    e.g.: Parent = 5 7 | 1 3 6 4 | 2
          Offspring = 5 7 | 4 6 3 1 | 2
'''
def mutation(populations, V):
    offspring = []
    heapq.heapify(offspring)
    population = copy.deepcopy(populations)
    for pop in population:
        k = randint(0,V-1) 
        l = randint(0, V-1)
        while k == l:
            l = randint(0, V-1)
        if l < k:
            k,l = l,k
        p = list(pop[1]) 

        while k <= l:
            p[k], p[l] = p[l], p[k]
            k+=1
            l-=1
        c1 = calculate_tsp_cost(p)
        if c1 != INF:
            heapq.heappush(offspring, tuple([1/c1, tuple(p)]))
        else:
            heapq.heappush(offspring, tuple([0, tuple(p)]))

    return offspring

'''
  PMX Crossover:
    1. A random split point is chosen
    2. Elements of the first half are exchanged as shown in the example
    Getting the first half same as other parent and mixing the properties of both in the 2nd half
    e.g.: P1: 5 7 1 | 3 6 4 2
          P2: 4 6 2 | 7 3 1 5

          Now we take node 5 (1st element) from P1 and swap its position with node 4 (1st element in P2) in P1 only ans same for Parent 2
          O1 = 4 7 1 | 3 6 5 2
          O2 = 5 6 2 | 7 3 1 4

          same is done till the partition

          Offspring1: 4 6 2 | 3 7 5 1 
          Offspring2: 5 7 1 | 6 3 2 4
'''
def pmx_crossover(populations, V):
    global graph
    offspring = []
    heapq.heapify(offspring)

    population = copy.deepcopy(populations)
    if(len(population) % 2 == 1):
        population.pop()
    
    for _ in range(0,len(population)//2):
        
        parent1 = heapq.heappop(population)
        p1 = list(parent1[1])
        parent2 = heapq.heappop(population)
        p2 = list(parent2[1])

        idx = randint(0, V-1)
        o1 = list(copy.deepcopy(p1))
        o2 = list(copy.deepcopy(p2))
        for j in range(0, idx+1):
            if p1[j] != p2[j]:
                ind1 = p1.index(p2[j])
                o1[j], o1[ind1] = o1[ind1], o1[j]
                ind2 = p2.index(p1[j])
                o2[j], o2[ind2] = o2[ind2], o2[j]
        
        for t in [o1, o2]:
            c1 = calculate_tsp_cost(t)
            if c1 != INF:
                heapq.heappush(offspring, tuple([1/c1, tuple(t)]))
            else:
                heapq.heappush(offspring, tuple([0, tuple(t)]))

    return offspring

'''
  Ordered Crossover:
    1. The parents are divided into 3 halves
    2. Middle segment of both is retained
    3. A sequence is generated starting from the last half and the elements of this seq not present in the offspring are appended
    4. Final sequence is inserted from the second cut point to generate 2 offsprings
    e.g.: P1 = 3 4 8 | 2 7 1 | 6 5
          P2 = 4 2 5 | 1 6 8 | 3 7

          seq1 = 6 5 3 4 8 2 7 1 => removing (1,6,8) => 5 3 4 2 7
          seq2 = 3 7 4 2 5 1 6 8 => removing (2,7,1) => 3 4 5 6 8
          
          offspring1 = 5 6 8 | 2 7 1 | 3 4
          offspring2 = 4 2 7 | 1 6 8 | 5 3
'''
def ordered_crossover(populations, V):
    global graph
    offsprings = []
    heapq.heapify(offsprings)
    population = copy.deepcopy(populations)

    if(len(population) % 2 == 1):
        population.pop()
    
    for _ in range(0,len(population)//2):
        p,q = V//3,(2*V)//3 
        
        p1 = heapq.heappop(population)
        parent1 = list(p1[1])
        p2 = heapq.heappop(population)
        parent2 = list(p2[1])

        o1 = [-1 for _ in range(V)]
        o2 = [-1 for _ in range(V)]
        
        for i in range(p+1,q+1):
            o1[i], o2[i] = parent1[i], parent2[i]

        i = (q+1) % V
        seq1 = []
        seq2 = []
        while i != q:
            seq1.append(parent2[i])
            seq2.append(parent1[i])
            i = (i+1) % V
        seq1.append(parent2[i])
        seq2.append(parent1[i])
        l = 0
        i = (q+1) % V
        while i != p+1:
            if seq1[l] not in o1: 
                o1[i] = seq1[l]
                i = (i+1) % V
            l += 1
        l = 0
        i = (q+1) % V
        while i != p+1:
            if seq2[l] not in o2: 
                o2[i] = seq2[l]
                i = (i+1) % V
            l += 1
        
        for t in [o1, o2]:
            c1 = calculate_tsp_cost(t)
        if c1 != INF:
            heapq.heappush(offsprings, tuple([1/c1, tuple(t)]))
        else:
            heapq.heappush(offsprings, tuple([0, tuple(t)]))

    return offsprings

'''
  Basic crossover:
    The first and second half of one parent is assigned to both the
    offsprings and parent 2 is used to fill the remaining nodes 
    e.g.: P1 = 3 4 8 2 | 7 1 6 5
          P2 = 4 2 5 1 6 8 3 7
          
          offspring1 = 3 4 8 2 | 5 1 6 7
          offspring2 = 7 1 6 5 | 4 2 8 3
'''
def normal_crossover(populations, V):
    global graph
    offsprings = []
    heapq.heapify(offsprings)

    population = copy.deepcopy(populations)
    
    if(len(population) % 2 == 1):
        heapq.heappop(population)
    
    for _ in range(0,len(population)//2):
        p1 = heapq.heappop(population)
        parent1 = list(p1[1])
        p2 = heapq.heappop(population)
        parent2 = list(p2[1])

        o1 = parent1[0:V//2]
        o2 = parent1[V//2:]
        for i in parent2:
            if i in parent1[V//2:]:
                o1.append(i)
            else:
                o2.append(i)

        for t in [o1, o2]:
            c1 = calculate_tsp_cost(t)
        if c1 != INF:
            heapq.heappush(offsprings, tuple([1/c1, tuple(t)]))
        else:
            heapq.heappush(offsprings, tuple([0, tuple(t)]))

    return offsprings

# Util function to generate all offsprings
def generate_children(population, V, population_size):
  # All functions generate offspring equal to population_size
  
    off1 = normal_crossover(population, V)   
    off2 = pmx_crossover(population, V) 
    off3 = ordered_crossover(population, V)  
    mut_pop = mutation(population, V)
    
    for i in off1:
        population.append(i)
    for i in off2:
        population.append(i)
    for i in off3:
        population.append(i)
    for i in mut_pop:
        population.append(i)

    # Retaining the best k offsprings from all the ones created
    for _ in range(len(population) - population_size):
        heapq.heappop(population)
    
    return population

# Driver function to calculate optimal path using genetic algorithm
def find_optimal_tsp_path(V, population, population_size=1000, no_generations = 200):
    global graph
    current_generation = 1
    optimal_tsp_path = list()
    optimal_tsp_cost = -1

    optimal_tsp_cost_array = []
    
    try:
        while current_generation < no_generations:
            print("\n--------- Current generation: ", current_generation, "---------")
            pop = list(item for item in population)

            min_cost = 0
            tsp_path = []
            for i in pop:
                if i[0] >= min_cost:
                    min_cost = i[0]
                    tsp_path = i[1]

            # Checking optimal path and cost till now
            if optimal_tsp_cost <= min_cost:
                optimal_tsp_cost = min_cost
                optimal_tsp_path = tsp_path
            print("COST: ", (1/optimal_tsp_cost), " PATH: ", optimal_tsp_path)

            optimal_tsp_cost_array.append(1/optimal_tsp_cost)
    
            # generating population for next generation
            population = generate_children(population, V, population_size)

            current_generation += 1

        return optimal_tsp_cost_array, optimal_tsp_path, current_generation
    except KeyboardInterrupt:
        return optimal_tsp_cost_array, optimal_tsp_path, current_generation

# To check if graph is connected
def dfs(visited, source):
    global graph
    if visited[source] == True:
        return
    visited[source] = True
    for neighbours in graph[source]:
        dfs(visited, neighbours)
        
def check_connected(source, V):
    global graph
    visited = [False] * V
    dfs(visited, source)
    
    j = 0
    for i in visited:
        if i == False:
            return False
        j+=1
    return True

# If the input file is in the lower diagonal form, just hardcode V = no of nodes in main function
# def lower_diag_input_util(filename):
#     global graph
#     numbers = list()
#     with open(filename, "r") as given_file:
#         lines = given_file.readlines()
#         for i, line in enumerate(lines):
#             line = line.split()
#             for c in line:
#                 numbers.append(int(c))
#     nodes = 0
#     sz = len(numbers)
#     while nodes * (nodes + 1) / 2 != sz:
#         nodes += 1
#     adj = defaultdict(dict)
#     index = 0
#     for i in range(nodes):
#         for j in range(i + 1):
#             adj[i][j] = float(numbers[index])
#             adj[j][i] = float(numbers[index])
#             index += 1
#     graph = adj

if __name__ == '__main__': 

    # run pip install networkx if netwrokx module not installed for visualization of final path
    #store data values like a map key:value pair
    graph_input = None
    path = []
    population_size = 1000
    no_generations = 200
    inputFile = "inputs/input_5.txt"
    parser = argparse.ArgumentParser(description='Hyperparameters for genetic algorithm')
    parser.add_argument('--popsize', dest='popSize', type=int, help='Population size')
    parser.add_argument('--gen', dest='generations', type=int, help='Number of generations to run')
    parser.add_argument('--inputfile', dest='files', type=str, help='Input File for creating graph')
    args = parser.parse_args()
    
    if args.popSize:
        population_size = args.popSize
    if args.generations:
        no_generations = args.generations
    if args.files:
        inputFile = args.files

    f = open(inputFile,"r")
    lines = f.readline()
    V = int(lines)

    i = 0
    # input file is in the form of adjacency graph, so number of lines = total no of nodes in the graph

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
                    adding_new_edge_to_graph(i,j,float(c))
                j+=1
        i+=1

    # If input file is lower diagonal
    # V = 24
    # lower_diag_input_util("inputs/gr24_1272.txt")

    if len(graph) != V:
        i = len(graph)-1

        while i < count:
            adding_new_edge_to_graph(i,i,0)
            i+=1
    f.close
    if len(graph) != 0:

        if check_connected(list(graph.keys())[0], V):
          # Randomly generate initial population
            population = generate_population(V, population_size)
            
            # Find the path
            costs, path, gens = find_optimal_tsp_path( V, population, population_size, no_generations)
            final_cost = costs[-1]
            print("\n------------------- FINAL OUTPUT -------------------")
            print("FINAL PATH: ", path)
            print("FINAL COST: ", final_cost)

        else:
            print("\nTSP Not possible as graph not connected\n")

    else:
        print("\nGiven Graph is empty\n")

print("------------------- FINAL OUTPUT -------------------")
print("POPULATION SIZE: ", population_size, "\tTOTAL GENERATIONS: ", no_generations)
print("FINAL PATH: ", path)
print("FINAL COST: ", final_cost)
print("GENERATIONS: ", costs.index(final_cost), '\n')

plt.figure(figsize=(8, 6))
plt.title("Generations v/s TSP Cost")
plt.xlabel("Generaions")
plt.ylabel("TSP Cost")
plt.plot([i+1 for i in range(gens)], costs)
plt.show()

# Displaying original graph
G1=nx.Graph()
node_label = {}
font = {'size': 20}
plt.rc('font', **font)
plt.figure(figsize=(6, 6))
plt.title("Graph along with TSP path")
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

'''
References Used:
  1. Ordered Crossover: https://www.hindawi.com/journals/cin/2017/7430125/
  2. PMX Crossover: https://user.ceng.metu.edu.tr/~ucoluk/research/publications/tspnew.pdf
'''