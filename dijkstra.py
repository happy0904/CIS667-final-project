import collections
import heapq
import numpy as np
class Cheapest_Path_searching_dijkstra:
    def __init__(self,size,G,G1,distance):
        #distance is 2D array, distance[start][end]=Euclidean distance between two cities.
        #self.heuristic=heuristic
        #risk[v]=the minimum cost of all pathes that via v
        self.risk = [float('inf') for i in range(size)]
        #path_cost[v]=the smallest cost from start city to v
        self.path_cost=[float('inf') for i in range(size)]
        self.goal = float('inf')
        #parent[v]=parent of city v, i.e. previous city before v
        self.parent=[None for i in range( size)]
        self.distance=distance
        #G is the graph G(V,E), which is a list of tuple, each tuple =(c,i,j) means there is direct path from
        #city i to city j, and the cost is 'c'.
        self.G=G
        self.G1=G1
        self.count=0

    def print_path(self, start,e):
        if self.parent[e]==None:
            print(e) if e==start else print('No path')
            return
        self.print_path(start,self.parent[e])
        print(self.parent[e],e)


    def sps_domain(self, start, end):
        self.path_cost[start]=0
        Open=[(0,start)]

        heapq.heapify(Open)
        Closed=[]
        Graph=collections.defaultdict(list)
        for s,e,c in self.G:
            Graph[s].append((e,c))
            Graph[e].append((s,c))
        self.risk[start]=self.path_cost[start]+self.G1[start][end]
        while Open:
            cost,v=heapq.heappop(Open)
            self.count+=1
            print('dijkstra current check node {}, current best price is {}'.format(v, self.goal))
            Closed.append(v)
            if v==end and self.path_cost[v]<self.goal:
                self.goal=self.path_cost[v]
                self.parent[end]=self.parent[v]
            #check each child node of parent node v
            for d,cost in Graph[v]:
                # if child node not visited or find a better path, replace previous one and record the new route
                if d not in Closed and self.path_cost[d]>self.path_cost[v]+cost:
                    self.path_cost[d]=self.path_cost[v]+cost
                    self.risk=self.path_cost[d]+self.G1[d][end]
                    self.parent[d]=v
                    heapq.heappush(Open,(self.path_cost[d],d))
        print('Cheapest path found with Dijkstra', self.goal)
        return -1
    def node_count(self):
        print('Num of node searched dijkstra',self.count)

def E_distance(x1,y1,x2,y2):
    r=(x1-x2)**2+(y1-y2)**2
    return np.sqrt(r)











