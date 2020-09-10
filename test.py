# class Solution:
#     def exist(self, board, word) -> bool:
#         stack = []
#         # visited = [0 for i in range(board) for j in range(board[0])]
#         # print(visited)
#         for i in range(len(board)):
#             for j in range(len(board[0])):
#                 if board[i][j] == word[0]:
#                     stack.append(i)
#                     stack.append(j)
#         print(stack)
#         if not stack:
#             return False
#         k = 1

#         while stack:
#             j = stack.pop()
#             i = stack.pop()
#             # print(stack)

#             if k==len(word):
#                 return True

#             board[i][j] = -1
#             if i-1>=0 and board[i-1][j]==word[k] and board[i-1][j]!=-1: #search up
#                 k += 1
#                 stack.append(i-1)
#                 stack.append(j)
#             elif i+1<len(board) and board[i+1][j]==word[k] and board[i+1][j]!=-1: #search down
#                 k += 1
#                 stack.append(i+1)
#                 stack.append(j)
#             elif j-1>=0 and board[i][j-1]==word[k] and board[i][j-1]!=-1: #search left
#                 k += 1
#                 stack.append(i)
#                 stack.append(j-1)
#             elif j+1<len(board[0]) and board[i][j+1]==word[k] and board[i][j+1]!=-1: #serch right
#                 k += 1
#                 stack.append(i)
#                 stack.append(j+1)
#             print(k)
            
#             print(stack)
#         return False
# '''
# A B C E
# S F C S
# A D E E 
# '''
# so = Solution()
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = "SEE"

# # board = [["a","b"],["c","d"]]
# # word = "abcd"

# # board = [["a","b","c","e"],
# # ["s","f","c","s"],
# # ["a","d","e","e"]]
# # word = "bfce"
# # board = [["a", "a"]]
# # word ="a"
# print(so.exist(board, word))

# import queue
# class Solution:
#     def movingCount(self, m: int, n: int, k: int) -> int:
#         visited = set()
#         que = queue.Queue()
#         que.put((0,0))

#         while que:
#             x, y = que.get()
#             if (x,y) not in visited and self.getsum(x,y)<=k:
#                 visited.add((x,y))
#                 # print(len(visited))
#                 if 0<=x+1<m:
#                     que.put((x+1,y))
#                 if 0<=y+1<n:
#                     que.put((x,y+1))

#         return len(visited)
    
#     def getsum(self, row, clom):
#         sums = 0
#         while row>0:
#             sums += row%10
#             row = row//10
        
#         while clom>0:
#             sums += clom%10
#             clom = clom//10
        
#         return sums

import collections
class Solution:
    def sum_rc(self,row,col):
        tmp = 0
        while row > 0:
            tmp += row % 10
            row //= 10
        while col > 0:
            tmp += col % 10
            col //= 10
        return tmp

    def movingCount(self, m: int, n: int, k: int) -> int:
        marked = set()  # 将访问过的点添加到集合marked中,从(0,0)开始
        queue = collections.deque()
        queue.append((0,0))
        while queue:
            x, y = queue.popleft()
            if (x,y) not in marked and self.sum_rc(x,y) <= k:
                marked.add((x,y)) 
                for dx, dy in [(1,0),(0,1)]:  # 仅考虑向右和向下即可
                    if 0 <= x + dx < m and 0 <= y + dy < n:
                        queue.append((x+dx,y+dy)) 
        return len(marked)

def getsum(row, clom):
    sums = 0
    while row>0:
        sums += row%10
        row = row//10
        
    while clom>0:
        sums += clom%10
        clom = clom//10
    return sums
print(getsum(35,37))
import queue
que = queue.Queue()
que.put((3,5))
que.put((2,6))
x,y = que.get()
print(que.get())
print(x, y)
a =Solution()
print(a.movingCount(2,3,1))