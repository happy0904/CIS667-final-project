# CIS667-final-project
cheapest flight tickets search
Didn't use anyone's existing code, just followed some similar strategies from our hw codes.
NN_plot.py is for plotting the loss trend
main.py is include all the function need to run: Neuron network training, Cheapest ticket search, and data generation.(Just need to run this one, since I import NN_plot to this file.
Size: number of different cities. (The only parameter needed to change for test, all the rest parameters depend on size)
Coordinate: coordinates of each city in 2D plane
flights: list of pair of cities that have direct flights
distance: 2D array, distance[i][j]=distance between city i and city j
G: 2D array, G[i]=[flight[i+j][0],flight[i+j][1], price(i,j)]].Assume if there are flights from i to j and j to i.
start: start city
end:end city
Input: Size, G, distance, start, end
Output: suboptimal path, suboptimal price
I claim all the coding finished myself.
