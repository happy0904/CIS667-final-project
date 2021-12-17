# CIS667-final-project
cheapest flight tickets search
Didn't use anyone's existing code, just followed some similar strategies from our hw codes.

Tool need to intall:
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

from NN_Plot import NN_plot
from sub_optimal import Cheapest_Path_searching_baseline_AI
from dijkstra import Cheapest_Path_searching_dijkstra

NN_plot.py is for plotting the loss trend
sub_optimal is A* searching without NN
final_test.py is include all the function need to run: Neuron network training, Cheapest ticket search, and data generation.
When you start to run main.py, there will be prompt :'Enter you size value:' show up, once you enter a integer number, the program will run
(I suggest start from smaller number first, say 5, then increase to 10,20, etc. It may be slow for very larger number). After several seconds...(for size=30, it will take less than 1 min)

Then there will be prompt ask for choose which algorithm you want to test, there are four options:Dijkstra,baseline_AI, NN_AI, Both.(Both means all the algorithm will run at the same time, since my data are generated randomly, if you run each algorithm seperately, you may have different problems everytime, it's hard to compare, but you can run each of them seperately first, then run 'Both'.
Meaning of each parameter:
Size: number of different cities. (The only parameter needed to change for test, all the rest parameters depend on size, will be generated randomly)
Coordinate: coordinates of each city in 2D plane
flights: list of pair of cities that have direct flights
distance: 2D array, distance[i][j]=distance between city i and city j
G1:2D array, if i,j connected, then G1[i][j]=tickets price, else inf
G: 2D array, G[i]=[flight[i+j][0],flight[i+j][1], price(i,j)]].Assume if there are flights from i to j and j to i.
start: start city
end:end city
Input: Size, G, distance, start, end
Output: nodes count by Dijkstra,A*NN_A*, optimal path(from dijkstra), optimal price, suboptimal path (from A*), suboptimal price
plot:loss trend, approximate value of optimal heuristic vs sub_optimal heuristic
I claim all the coding finished myself.

Citations of library:
https://numpy.org/citing-numpy/
https://doi.org/10.5281/zenodo.4724125
https://doi.org/10.1109/MCSE.2007.55
Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
