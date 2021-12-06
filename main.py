# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
class Cheapest_Path_searching_domain:
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
            risk=min(risk,self.path_cost[v]+self.optimal_heuristic(v,end))
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

            Closed.append(v)
            if v==end and self.path_cost[v]<self.goal:
                self.goal=self.path_cost[v]
                self.parent[end]=self.parent[v]
            if self.goal<a*self.min_cost(Open,end):
                print('Cheapest path found', self.goal, self.min_cost(Open,end))
                return
            #check each child node of parent node v
            for d,cost in Graph[v]:
                # if child node not visited or find a better path, replace previous one and record the new route
                if d not in Closed or self.path_cost[d]>self.path_cost[v]+cost:
                    self.path_cost[d]=self.path_cost[v]+cost
                    self.risk=self.path_cost[d]+self.optimal_heuristic(d,end)
                    self.parent[d]=v
                    heapq.heappush(Open,(self.path_cost[d],d))
        return -1


#test part

size=random.randint(4,20)#num of different city
n=int(size*(size-1)/2.0)#num of combinations of different pair of cities.
C=list(combinations(np.arange(size),2))
cost=random.sample(range(100, 3000), n)

start,end=random.sample(range(size), 2)#start and end city

flights=list(set(random.choices(C,k=int(size))))
f=len(flights)
#G=[[0,1,0],[0,2,0],[2,3,0],[1,4,0]]
G=[[0]*3 for i in range(2*f)]
#not all pair of cities have direct flight,randomly choose k=size pair of cities,
#assume that each picked pair of cities have flights to and from each other
for i in range(f):
    G[i][0],G[i][1]=flights[i]
    G[f+i][0],G[f+i][1]=flights[i][1],flights[i][0]
#print(G)

distance=[[0]*size for i in range(size)]
#initialize distance between any pair of cities
for i,(s,e) in enumerate(C):
    distance[e][s]=distance[s][e]=cost[i]
#Initialize tikect price for any pair of cities that have direct flight
for i in range(f):
    s,e=G[i][0],G[i][1]
    #If distance between two city is less than 500, then the tickect price is 100+a random num in range(0,50)
    ran=random.uniform(0,50)
    G[f+i][2]=G[i][2]=100+ran if distance[s][e]<500 else 100+(distance[s][e]-500)*0.3+ran

domain=Cheapest_Path_searching_domain(size,G,distance)
domain.sps_domain(start,end,1)
print("G",G)
print('Distance',distance)
print('Cheapest path for start city {}, end city {} with price {}'.format(start,end, domain.goal))
domain.print_path(end)






"""
size=5
n=int(size*(size-1)/2.0)
arr=np.arange(5)

C=list(combinations(arr,2))
cost=random.sample(range(100, 3000), n)

distance=[[0]*size for i in range(size)]
G=[[0,1,0],[0,2,0],[2,3,0],[1,4,0]]
start=0
end=4
#initialize distance between any pair of cities
for i,(s,e) in enumerate(C):
    distance[e][s]=distance[s][e]=cost[i]
#Initialize tikect price for any pair of cities that have direct flight
for i in range(len(G)):
    s,e=G[i][0],G[i][1]
    #If distance between two city is less than 500, then the tickect price is 100+a random num in range(0,50)
    ran=random.uniform(0,50)
    G[i][2]=100+ran if distance[s][e]<500 else 100+(distance[s][e]-500)*0.3+ran
domain=Shortest_Path_searching_domain(size,G,distance)
domain.sps_domain(start,end,0.1)
print(G)
print(distance)
domain.print_path(end)
"""



















# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
