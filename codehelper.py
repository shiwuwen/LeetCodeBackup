while True:
    try:
        temp = list(map(int, input().strip().split()))
    except:
        break

nums = int(input().strip())
for _ in range(nums):
    temp = list(map(int, input().strip().split()))
    res = 0
    if temp[0] > 0:
        for i in range(temp[0]):
            res += temp[i+1]
    print(res)

'''
#最小值堆
import heapq

a = [2, 5, 1, 0, 18]
heapq.heapify(a)
print(a)
temp = heapq.heappop(a)
print(temp)
heapq.heappush(a, -1)
print(a[0])
'''

'''
#最小生成树和最短路径
def prim(graph):
    m = len(graph)
    parent = [0 for _ in range(m)]
    lowestcost = [val for val in graph[0]]
    res = []

    for index in range(1, m):
        min = 1000
        k = 0
        for j in range(1, m):
            if lowestcost[j] != 0 and lowestcost[j] < min:
                min = lowestcost[j]
                k = j
        
        lowestcost[k] = 0
        res.append([parent[k], k])

        for l in range(1, m):
            if lowestcost[l] != 0 and graph[k][l] < lowestcost[l]:
                lowestcost[l] = graph[k][l]
                parent[l] = k
    
    return res

def dijkstra(graph):
    m = len(graph)
    parent = [0 for _ in range(m)]
    lowestcost = [val for val in graph[0]]
    visited = [0 for _ in range(m)]
    visited[0] = 1

    for _ in range(1, m):
        min = 10000
        k = 0
        for j in range(m):
            if visited[j] != 1 and lowestcost[j] < min:
                min = lowestcost[j]
                k = j
        
        visited[k] = 1

        for l in range(m):
            if visited[l] != 1 and lowestcost[k] + graph[k][l] < lowestcost[l]:
                lowestcost[l] = graph[k][l] + lowestcost[k]
                parent[l] = k
    
    print(parent)
    return lowestcost

graph = [[0, 10, 1000, 1000, 1000, 11, 1000, 1000, 1000],
         [10, 0, 18, 1000, 1000, 1000, 16, 1000, 12],
         [1000, 1000, 0, 22, 1000, 1000, 1000, 1000, 8],
         [1000, 1000, 22, 0, 20, 1000, 1000, 16, 21],
         [1000, 1000, 1000, 20, 0, 26, 1000, 7, 1000],
         [11, 1000, 1000, 1000, 26, 0, 17, 1000, 1000],
         [1000, 16, 1000, 1000, 1000, 17, 0, 19, 1000],
         [1000, 1000, 1000, 16, 7, 1000, 19, 0, 1000],
         [1000, 12, 8, 21, 1000, 1000, 1000, 1000, 0]]

result = dijkstra(graph)
print(result)
'''