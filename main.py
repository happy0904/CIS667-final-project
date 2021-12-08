# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
#NN_plot plots the loss trendency
from NN_Plot import NN_plot
#neuron network for heuristic training
class NN_heuristic:
    #x,y training set. xTest,yTest testing set. Coordinates of all the cities.
    def __init__(self,x, y, Input, xTest, yTest, size):
        self.x=x
        self.y=y
        self.xTest=xTest
        self.yTest=yTest
        self.size=size

    def train(self, v, end):
        scale_x = MinMaxScaler()
        x = scale_x.fit_transform(self.x)
        xTest = scale_x.fit_transform(self.xTest)
        scale_y = MinMaxScaler()
        y = scale_y.fit_transform(self.y)
        yTest = scale_y.fit_transform(self.yTest)
        model = keras.Sequential()
        model.add(layers.Dense(8, input_dim=4, activation="relu", kernel_initializer='he_uniform'))
        model.add(layers.Dense(5, activation="sigmoid", kernel_initializer='he_uniform'))
        model.add(layers.Dense(8, activation="relu", kernel_initializer='he_uniform'))
        model.add(layers.Dense(1))
        model.compile(loss='mse', optimizer='adam',metrics=['acc'])
        P = model.fit(x, y, epochs=500, batch_size=5, verbose=0)
        #P=model.fit(x, y, epochs=500, validation_data = (xTest, yTest),batch_size=5, verbose=0)
        #yPred = model.predict(x)
        """
        yT=model.predict(xTest)
        #ynew=scale_y.inverse_transform(ynew)
        train_loss = P.history['loss']
        test_loss = P.history['val_loss']
        train_acc = P.history['acc']
        test_acc = P.history['val_acc']
        xc = range(500)
        plt.figure()
        plt.plot(xc, train_loss)
        plt.plot(xc, test_loss)

        plt.show()

        x = scale_x.inverse_transform(x)
        yPred = scale_y.inverse_transform(yPred)
        y = scale_y.inverse_transform(y)
        #yPred = scale_y.inverse_transform(yPred)
        plt.plot(x, y, 'r+', x, yPred, 'g.')
        plt.xlabel('$X$')
        plt.ylabel('$Pred(x)$')
        plt.grid(True)

        plt.legend(['green:pred'], loc='upper left')
        #plt.plot(P.history['acc'])
        plt.show()
        """
        xnew=np.array(Input[v*size+end])
        xnew = np.reshape(xnew, (1, -1))
        ynew = model.predict(xnew)
        ynew = scale_y.inverse_transform(ynew)
        return ynew



class Cheapest_Path_searching_domain:
    def __init__(self,size,G,distance,T_heuristic):
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
        self.T_heuristic=T_heuristic


    def optimal_heuristic(self,start,end):
        #optimal_heuristic by intuition is the cost calculated wrt the shortest distance between two city
        return 100 if self.distance[start][end]<500 else 100+0.3*(self.distance[start][end]-500)
        #self.distance[start][end]

    def print_path(self, e):
        if self.parent[e]==None:
            print(e) if e==start else print('No path')
            return
        self.print_path(self.parent[e])
        print(self.parent[e],e)

    def min_cost(self, P, end):
        risk=float('inf')
        for _,v in P:
            risk=min(risk,self.path_cost[v]+self.T_heuristic(v,end))#self.optimal_heuristic(v,end)
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

            Closed.append(v)
            if v==end and self.path_cost[v]<self.goal:
                self.goal=self.path_cost[v]
                self.parent[end]=self.parent[v]
            if self.goal<a*self.min_cost(Open,end):
                print('Cheapest path found ', self.goal)
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
        print('num of node',self.count)

def E_distance(x1,y1,x2,y2):
    r=(x1-x2)**2+(y1-y2)**2
    return np.sqrt(r)


#test part

size=15#random.randint(4,20)#num of different city
n=int(size*(size-1)/2.0)#num of combinations of different pair of cities.
C=list(combinations(np.arange(size),2))
Coordinate=[3000*np.random.random_sample(size = 2) for i in range(size)]
start,end=random.sample(range(size), 2)
distance=np.array([[0]*size for i in range(size)],dtype=float)

for i in range(size):
    for j in range(i+1,size):
        # distance between city i and city j should be same.
        # distance[i][i]=0
        x1,y1,x2,y2=Coordinate[i][0],Coordinate[i][1],Coordinate[j][0],Coordinate[j][1]
        distance[j][i]=distance[i][j]=E_distance(x1,y1,x2,y2)

flights=list(set(random.choices(C,k=int(max(size,n/4.0)))))
f=len(flights)
G=[[0]*3 for i in range(2*f)]
#not all pair of cities have direct flight,randomly choose k pair of cities,
#assume that each picked pair of cities have flights to and from each other
for i in range(f):
    G[i][0],G[i][1]=flights[i]
    G[f+i][0],G[f+i][1]=flights[i][1],flights[i][0]
#Initialize tikect price for any pair of cities that have direct flight
for i in range(f):
    s,e=G[i][0],G[i][1]
    #If distance between two city is less than 500, then the tickect price is 100+a random num in range(0,50)
    ran=random.uniform(0,50)
    G[f+i][2]=G[i][2]=100+ran if distance[s][e]<500 else 100+(distance[s][e]-500)*0.2+ran
Input=np.array([[0]*4 for i in range(size**2)])
Output=np.array([0 for i in range(size**2)])
for i in range(size):
    for j in range(size):
        Input[j+size*i]=[Coordinate[i][0],Coordinate[i][1],Coordinate[j][0],Coordinate[j][1]]
        Output[j+size*i]=100 if distance[i][j]<500 else 100+(distance[i][j]-500)*0.2
batchSize = int(size ** 2 / 2)
#Generation of trainning set and testing set of NN model
Input = np.reshape(Input, (Input.shape[0], -1))
num_train=np.random.choice(np.arange(size**2),batchSize)
num_test=np.random.choice(np.arange(size**2),batchSize)

Output = np.reshape(Output, (Output.shape[0], -1))
xBatch=Input[num_train]
yBatch=Output[num_train]
#print('xBatch',xBatch)
#print('yBatch',yBatch)
xTest=Input[num_test]
yTest=Output[num_test]
Test=NN_heuristic(xBatch,yBatch,Input,xTest,yTest,size)
#T_heuristic=np.array([0 for i in range(size**2)])
#T_heuristic=np.reshape(T_heuristic, (T_heuristic.shape[0], -1))
NN=NN_plot(xBatch,yBatch,Input,xTest,yTest,size)
T_heuristic_plot=NN.train()
domain=Cheapest_Path_searching_domain(size,G,distance,T_heuristic=Test.train)
domain.sps_domain(start,end,1.0)
print('flight',flights)
print("G",G)
print('Cheapest path for start city {}, end city {} with price {}'.format(start,end, domain.goal))
domain.print_path(end)
domain.node_count()












# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
