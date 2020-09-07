class Solution:
    def exist(self, board, word) -> bool:
        stack = []
        # visited = [0 for i in range(board) for j in range(board[0])]
        # print(visited)
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]:
                    stack.append(i)
                    stack.append(j)
        print(stack)
        if not stack:
            return False
        k = 1

        while stack:
            j = stack.pop()
            i = stack.pop()
            # print(stack)

            if k==len(word):
                return True

            board[i][j] = -1
            if i-1>=0 and board[i-1][j]==word[k] and board[i-1][j]!=-1: #search up
                k += 1
                stack.append(i-1)
                stack.append(j)
            elif i+1<len(board) and board[i+1][j]==word[k] and board[i+1][j]!=-1: #search down
                k += 1
                stack.append(i+1)
                stack.append(j)
            elif j-1>=0 and board[i][j-1]==word[k] and board[i][j-1]!=-1: #search left
                k += 1
                stack.append(i)
                stack.append(j-1)
            elif j+1<len(board[0]) and board[i][j+1]==word[k] and board[i][j+1]!=-1: #serch right
                k += 1
                stack.append(i)
                stack.append(j+1)
            print(k)
            
            print(stack)
        return False
'''
A B C E
S F C S
A D E E 
'''
so = Solution()
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
word = "SEE"

# board = [["a","b"],["c","d"]]
# word = "abcd"

# board = [["a","b","c","e"],
# ["s","f","c","s"],
# ["a","d","e","e"]]
# word = "bfce"
# board = [["a", "a"]]
# word ="a"
print(so.exist(board, word))