"""
    Name: Vanshika Jain
    Roll No.: BT18CSE107
    Algorithm Implemented: Bi-Directional BFS
"""
from collections import deque
import heapq
import numpy as np
import random
import sys

N = 3
# Class to define the tile
class Tile:

  # List to store all possible directions of dx and dy namely
  # (0,-1), (-1,0), (0,1) and (1,0)
  dirs = [0, -1, 0, 1, 0]
  
  def __init__(self, state, parent = None):
      self.state = state
      self.parent = parent

# Find set of all the valid moves from the current state
  def getMoves(self):
    moves = []
    index = self.state.index(0)
    x = index % N
    y = index // N
    for i in range(4):
      tx = x + Tile.dirs[i]
      ty = y + Tile.dirs[i + 1]
      if tx < 0 or ty < 0 or tx == N or ty == N:
        continue
      i = ty * N + tx
      move = list(self.state)
      move[index] = move[i]
      move[i] = 0
      moves.append(tuple(move))
    return moves

# Print current node
  def printNode(self):
    mat = np.reshape(self.state, (N, N))
    for i in range(N):
        for j in range (N):
            print(mat[i][j], end=" ")
        print()

def getRootTile(n):
  return getRootTile(n.parent) if n.parent else n

"""
  We are traversing from both the sides so we have 2 root nodes, one is the 
  start state and other is the goal state. We want a single path finally, 
  so we reverse the pointers so as to get a single top to down path
"""
def constructPath(p, o):
    while o:
      t = o.parent
      o.parent = p
      p, o = o, t
    return p

def BidirectionalBFS(start_state, goal_state):
  ns = Tile(start_state)
  ne = Tile(goal_state)
  q = [deque([ns]), deque([ne])]  #Queue to traverse
  fringeList = [{start_state : ns}, {goal_state: ne}]  # List to append elements that are traversed
  closed = 0  #Variable to keep track of nodes that were explored
  f = 0   #Flag to decide which side traversal

  """
      Doing BFS turn by turn
      First top to bottom then vice versa
  """
  while q[f]:
    l = len(q[f])
    while l > 0:
      p = q[f].popleft()
      closed += 1
      # Find all the successor nodes and iterating over them
      for move in p.getMoves():
        n = Tile(move, p)
        if move in fringeList[1-f]:
          o = fringeList[1-f][move]
          if getRootTile(n).state == goal_state:
            o, n = n, o

          n = constructPath(n, o.parent)
          return n, len(fringeList[f]) + len(fringeList[1-f]), closed

        if move in fringeList[f]: continue

        fringeList[f][move] = n
        q[f].append(n)
      l -= 1
    f = 1 - f

  return None, len(fringeList[0]) + len(fringeList[1]), closed

"""
    Function to compute path from the start state
    to the goal state after a central node is
    reached from both the sides
"""
def print_path(n):
  if not n: return
  print_path(n.parent)
  n.printNode()
  print("\n------------------------\n")

"""
    Parity test to check the number of inverions
    and make sure that we can reach the goal state
    from the given start state

    If inversion is even then it is possible to
    convert else it is not
"""
def solvable(state):
  inv = 0
  for i in range(N*N):
    for j in range(i + 1, N*N):
      if all((state[i] > 0, state[j] > 0, state[i] > state[j])):
        inv += 1
  return inv % 2 == 0

if __name__ == '__main__':
  # start state used to explain in video
  # start_state = [0, 8, 7, 6, 5, 4, 3, 2, 1] 
  start_state =[2, 1, 5, 3, 0, 6, 4, 7, 8]
  goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]

# We can create random start states and if we encounter a state which can be solved, 
# BidirectionalBFS is called on it
  while True:
    random.shuffle(start_state)
    if solvable(start_state):
      break
  
  if solvable(start_state):
    n, fringeList, closed = BidirectionalBFS(tuple(start_state), tuple(goal_state))

    print("------GIVEN START STATE-------\n")
    Tile(start_state).printNode()

    print("\n\n------Bidirectional BFS Steps-------\n")
    print_path(n)

    print("-------EXPECTED GOAL STATE-------\n")
    Tile(goal_state).printNode()

    print("Total nodes explored ", fringeList)
    print("Total nodes closed after which we got the solution ", closed)
