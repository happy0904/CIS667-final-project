# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import combinations
import collections
import heapq
import numpy as np
import os
import tensorflow as tf

# tf.random.set_seed(7)
import scipy.special as sp
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler




class Cheapest_Path_searching_baseline_AI:
    def __init__(self,size,G,distance):
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
        self.count=0



    def optimal_heuristic(self,start,end):
        #optimal_heuristic by intuition is the cost calculated wrt the shortest distance between two city
        return 100 if self.distance[start][end]<500 else 100+0.2*(self.distance[start][end]-500)
        #self.distance[start][end]

    def print_path(self, start,e):
        if self.parent[e]==None:
            print(e) if e==start else print('No path')
            return
        self.print_path(start,self.parent[e])
        print(self.parent[e],e)

    def min_cost(self, P, end):
        risk=float('inf')
        for _,v in P:
            risk=min(risk,self.path_cost[v]+self.optimal_heuristic(v,end))#self.optimal_heuristic(v,end)
        return risk

    def sps_domain(self, start, end, a):
        self.path_cost[start]=0
        Open=[(0,start)]

        heapq.heapify(Open)
        Closed=[]
        Graph=collections.defaultdict(list)
        for s,e,c in self.G:
            Graph[s].append((e,c))
            Graph[e].append((s,c))
        self.risk[start]=self.path_cost[start]+self.optimal_heuristic(start,end)
        while Open:
            cost,v=heapq.heappop(Open)
            self.count+=1
            print('baseline_AI current check node {}, current best price is {}'.format(v, self.goal))
            Closed.append(v)
            if v==end and self.path_cost[v]<self.goal:
                self.goal=self.path_cost[v]
                self.parent[end]=self.parent[v]
            if self.goal<a*self.min_cost(Open,end):
                print('Cheapest path found with baseline_AI', self.goal)
                return
            #check each child node of parent node v
            for d,cost in Graph[v]:
                # if child node not visited or find a better path, replace previous one and record the new route
                if d not in Closed and self.path_cost[d]>self.path_cost[v]+cost:
                    self.path_cost[d]=self.path_cost[v]+cost
                    self.risk=self.path_cost[d]+self.optimal_heuristic(d,end)
                    self.parent[d]=v
                    heapq.heappush(Open,(self.path_cost[d],d))
        return -1
    def node_count(self):
        print('Num of node searched with baseline_AI',self.count)

def E_distance(x1,y1,x2,y2):
    r=(x1-x2)**2+(y1-y2)**2
    return np.sqrt(r)













#T_heuristic=np.array([0 for i in range(size**2)])
#T_heuristic=np.reshape(T_heuristic, (T_heuristic.shape[0], -1))

#print('T',T_heuristic)
#domain=Cheapest_Path_searching_domain(size,G,distance,T_heuristic=Test.train)
#domain.sps_domain(start,end,1.0)

#print('Cheapest path for start city {}, end city {} with price {}'.format(start,end, domain.goal))
#domain.print_path(end)
#domain.node_count()