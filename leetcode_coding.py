class minArray:
    def minArray(self, numbers):
        left = 0
        right = len(numbers) - 1

        if right == 0:
            return numbers[left]

        while left < right:
            mid = (left + right) // 2

            if mid > left and numbers[mid] < numbers[left] and numbers[mid] < numbers[right]:
                return numbers[mid]
            elif mid == left:
                return min(numbers[left], numbers[right])
            else:
                left = mid + 1
        
        return min(numbers[0], numbers[-1])

a = minArray()
numbers = [3,4,4]
print(a.minArray(numbers))

'''
在排序数组中查找数字 I

a = SearchNums()
nums = [7,7]
target = 7
print(a.search(nums, target))
'''
class SearchNums:
    def search(self, nums, target):
        left = 0
        right = len(nums)
        total_nums = 0

        if right == 0:
            return 0
        
        while left < right:
            mid = (left + right) // 2
            if target == nums[mid]:
                total_nums += 1
                temp = mid + 1
                while temp < right:
                    if nums[temp] == target:
                        total_nums += 1
                        temp += 1
                    else:
                        break
                
                temp = mid - 1
                while temp >= left:
                    if nums[temp] == target:
                        total_nums += 1
                        temp -= 1
                    else:
                        break
                left = right + 1
            elif target < nums[mid]:
                right = mid
            else:
                left = mid + 1
            
        return total_nums


'''
包含min函数的栈
'''
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = collections.deque()
        self.min_stack = collections.deque()


    def push(self, x: int) -> None:
        self.stack.append(x)
        if not self.min_stack:
            self.min_stack.append(x)
        elif x <= self.min_stack[-1]:
            self.min_stack.append(x)


    def pop(self) -> None:
        temp = self.stack.pop()

        if temp == self.min_stack[-1]:
            self.min_stack.pop()


    def top(self) -> int:
        return self.stack[-1]


    def min(self) -> int:
        return self.min_stack[-1]


'''
用两个栈实现队列
'''
class CQueue:

    def __init__(self):
        self.left = collections.deque()
        self.right = collections.deque()


    def appendTail(self, value: int) -> None:
        self.left.append(value)

    def deleteHead(self) -> int:
        if not self.left:
            return -1
        
        if self.left:
            while self.left:
                self.right.append(self.left.pop())
            
            temp = self.right.pop()
            while self.right:
                self.left.append(self.right.pop())
            
            return temp


'''
罗马数字转整数
s = "MCMXCIV" 1994
a = romanToIntSolution()
s = "CMXCIX" # 999
print(a.romanToInt(s))
'''
class romanToIntSolution:
    def romanToInt(self, s):
        total_num = 0

        num_map = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}

        length = len(s)
        if length == 1:
            return num_map[s[0]]
        
        index = 0

        while index < length:
            if s[index] != 'I' and s[index] != 'X' and s[index] != 'C':
                total_num += num_map[s[index]]
                index += 1
            elif s[index] == 'I':
                if index + 1 < length and (s[index+1] == 'V' or s[index+1] == 'X'):
                    total_num += num_map[s[index+1]] - num_map[s[index]]
                    index += 2
                else:
                    total_num += num_map[s[index]]
                    index += 1
            elif s[index] == 'X':
                if index + 1 < length and (s[index+1] == 'L' or s[index+1] == 'C'):
                    total_num += num_map[s[index+1]] - num_map[s[index]]
                    index += 2
                else:
                    total_num += num_map[s[index]]
                    index += 1
            elif s[index] == 'C':
                if index + 1 < length and (s[index+1] == 'D' or s[index+1] == 'M'):
                    total_num += num_map[s[index+1]] - num_map[s[index]]
                    index += 2
                else:
                    total_num += num_map[s[index]]
                    index += 1
        
        return total_num


'''
给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
solve(board)
print(board)
'''
def solve(board):
	# dfs
	# if not board:
	# 	return 

	# n = len(board)
	# m = len(board[0])

	# def dfs(x, y):
	# 	if not 0<=x<n or not 0<=y<m or board[x][y] != "O":
	# 		return

	# 	board[x][y] = "A"
		
	# 	dfs(x + 1, y)
	# 	dfs(x - 1, y)
	# 	dfs(x, y + 1)
	# 	dfs(x, y - 1)

	# for i in range(n):
	# 	dfs(i, 0)
	# 	dfs(i, m-1)

	# for j in range(m-1):
	# 	dfs(0, j)
	# 	dfs(n-1, j)
	
	# for i in range(n):
	# 	for j in range(m):
	# 		if board[i][j] == "A":
	# 			board[i][j] = "O"
	# 		elif board[i][j] == "O":
	# 			board[i][j] = "X"

	# bfs
	if not board:
		return 

	n = len(board)
	m = len(board[0])

	import collections
	queue = collections.deque()

	for i in range(n):
		if board[i][0] == 'O':
			board[i][0] = 'A'
			queue.append((i, 0))
		if board[i][m-1] == 'O':
			board[i][m-1] = 'A'
			queue.append((i, m-1))

	for j in range(1, m-1):
		if board[0][j] == 'O':
			board[0][j] = 'A'
			queue.append((0, j))
		if board[n-1][j] == 'O':
			board[n-1][j] = 'A'
			queue.append((n-1, j))

	while queue:
		x, y = queue.popleft()
		for i, j in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
			if 0<=i<n and 0<=j<m and board[i][j] == 'O':
				board[i][j] = 'A'
				queue.append((i, j))

	for i in range(n):
		for j in range(m):
			if board[i][j] == 'A':
				board[i][j] = 'O'
			elif board[i][j] == 'O':
				board[i][j] = 'X'


'''
给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
每条从根节点到叶节点的路径都代表一个数字：

例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
计算从根节点到叶节点生成的 所有数字之和 。

叶节点 是指没有子节点的节点。
'''
def sumNumbers(self, root):
    self.value = 0
    self.sum = 0
    def helper(root):
        if not root.left and not root.right:
            self.value = self.value * 10 + root.val
            self.sum += self.value
            self.value = (self.value - root.val) / 10
            return

        if root.left:
            self.value = self.value * 10 + root.val
            helper(root.left)
            self.value = (self.value - root.val) / 10

        if root.right:
            self.value = self.value * 10 + root.val
            helper(root.right)
            self.value = (self.value - root.val) / 10

    helper(root)
    return self.sum


'''
nums = [100,4,200,1,3,2]
nums = [0,3,7,2,5,8,4,6,0,1]
nums = [1,0,1,0]
print(longestConsecutive(nums))
'''
def longestConsecutive(nums):
    nums.sort()
    length = len(nums)

    if length <= 1:
        return length

    max_len = 0
    curr_len = 1

    for index in range(1, length):
        if nums[index] == nums[index-1]:
            max_len = max(max_len, curr_len)
        elif nums[index] == nums[index-1] + 1:
            curr_len += 1
        else:
            max_len = max(max_len, curr_len)
            curr_len = 1

    if nums[-1] == nums[-2] + 1:
        max_len = max(max_len, curr_len)

    return max_len


'''
给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
s = "A man, a plan, a canal: Panama"
s = "./"
print(isPalindrome(s))
'''
def isPalindrome(s):
	# something wrong when s = './'
    length = len(s)
    if length < 2:
        return True

    s = s.lower()

    left = 0
    right = length - 1

    while left < right and not s[left].isalnum():
        left += 1

    while right > left and not s[right].isalnum():
        right -= 1

    while left < right:
        if s[left] != s[right]:
            return False

        left += 1
        right -= 1

        while left < right and not s[left].isalnum():
            left += 1

        while right > left and not s[right].isalnum():
            right -= 1

    return False


'''
路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。
该路径 至少包含一个 节点，且不一定经过根节点。
路径和 是路径中各节点值的总和。
给你一个二叉树的根节点 root ，返回其 最大路径和 。
'''
class Solution:
    def __init__(self):
        self.max_sum = float('-inf')

    def maxPathSum(self, root):
        def maxSum(root):
            if not root:
                return 0

            leftSum = max(0, maxSum(root.left))
            rightSum = max(0, maxSum(root.right))

            currSum = root.val + leftSum + rightSum

            self.maxSum = max(self.maxSum, currSum)

            return root.val + max(leftSum, rightSum)

        maxSum(root)
        return self.maxSum


'''
给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

# prices = [3,3,5,0,0,3,1,4]
prices = [1,2,3,4,5]
print(maxProfit3(prices))
'''
def maxProfit3(prices):
    length = len(prices) + 1

    dp = [[[0 for j in range(2)] for k in range(3)] for i in range(length)]
    
    for i in range(length):
        dp[i][0][0] = 0
        dp[i][0][1] = -float('inf')

    for k in range(3):
        dp[0][k][0] = 0
        dp[0][k][1] = -float('inf')

    for index in range(1, length):
        for k in range(1, 3):
            dp[index][k][0] = max(dp[index-1][k][0], dp[index-1][k][1] + prices[index-1])
            dp[index][k][1] = max(dp[index-1][k][1], dp[index-1][k-1][0] - prices[index-1])

    return dp[-1][-1][0]


'''
输入: prices = [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
'''
def maxProfit2(prices):
    length = len(prices) + 1

    dp = [[0 for j in range(2)] for i in range(length)]
    dp[0][1] = -float('inf')

    for index in range(1, length):
        dp[index][0] = max(dp[index-1][0], dp[index-1][1] + prices[index-1])
        dp[index][1] = max(dp[index-1][1], dp[index-1][0] - prices[index-1])

    return dp[-1][0]


'''
给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。

prices = [7,1,5,3,6,4]
print(maxProfit(prices))
'''

def maxProfit(prices):
    length = len(prices) + 1

    dp = [[0 for j in range(2)] for i in range(length)]
    dp[0][1] = -float('inf')

    for index in range(1, length):
        dp[index][0] = max(dp[index-1][0], dp[index-1][1] + prices[index-1])
        dp[index][1] = max(dp[index-1][1], -prices[index-1])

    return dp[-1][0]


'''
给定一个三角形 triangle ，找出自顶向下的最小路径和。

每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。
也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。
triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
print(minimumTotal(triangle))
11
'''
def minimumTotal(triangle):
    length = len(triangle)
    # if length == 1:
        # return triangle[0][0]

    for index in range(length-2, -1, -1):
        curr = triangle[index]
        post = triangle[index + 1]

        for j in range(len(curr)):
            curr[j] = curr[j] + min(post[j], post[j+1])

    return triangle[0][0]


'''
给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。
print(generateTriangle(5))
'''
def generateTriangle(numRows):
    result = [[1]]
    if numRows < 2:
        return result
    result.append([1, 1])

    for index in range(2, numRows):
        length = len(result[-1])
        pre = result[-1]

        # curr = [1 for i in range(length+1)]
        curr = [1] * (length + 1) 
        mid = length // 2

        for j in range(1, mid+1):
            curr[j] = pre[j-1] + pre[j]
            curr[length - j] = curr[j]

        result.append(curr)

    return result


'''
给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。
'''
def connectTwoNode(self, node1, node2):
    if not node1 or not node2:
        return 

    node1.next = node2

    self.connectTwoNode(node1.left, node1.right)
    self.connectTwoNode(node2.left, node2.right)
    self.connectTwoNode(node1.right, node2.left)

def connect(self, root: 'Node') -> 'Node':
    if not root:
        return
    self.connectTwoNode(root.left, root.right)

    return root


'''
给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。
'''
def flatten(root):
    if not root:
        return

    self.flatten(root.left)
    self.flatten(root.right)

    left = root.left
    right = root.right

    root.right = left
    root.left = None

    temp = root
    while temp.right:
        temp = temp.right

    temp.right = right

'''
K个一组反转链表
'''
def reverseList(start, end):
    pre = None
    curr = net = start
    while curr != end:
        net = curr.next
        cur.next = pre
        pre = curr
        curr = net

    return pre 

def reverseKGroup(head, k):
    if not head:
        return head

    a = b = head

    for i in range(k):
        if not b:
            return head
        b = b.next

    newhead = reverseList(a, b)

    a.next = self.reverseKGroup(b, k)

    return newhead


'''
0-1 背包
weight = [2, 1, 3]
value = [4, 2, 3]
w = 5
print(zero_one_bags(value, weight, w))
print()
'''
def zero_one_bags(value, weight, w):
    length = len(value)

    dp = [[0 for i in range(w+1)] for j in range(length+1)]

    for i in range(1, length+1):
        for j in range(1, w+1):
            if j - weight[i-1] >= 0:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-weight[i-1]]+value[i-1])
            else:
                dp[i][j] = dp[i-1][j]

    return dp[-1][-1]


'''
给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。
子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。
s = "bbbab" 4
s = "cbbd" 2

a = Solution2()
print(a.longestPalindromeSubseq('cbbd'))
print()
'''
class Solution2:
    def longestPalindromeSubseq(self, s):
        length = len(s)

        dp = [[0 for j in range(length)] for i in range(length)]

        for i in range(length):
            dp[i][i] = 1

        for i in range(length-2, -1, -1):
            for j in range(i+1, length):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i][j-1], dp[i+1][j])

        return dp[0][length-1]


'''
最长递增子序列

'''
def LongestUpSubstring(input_list):
    length = len(input_list)

    dp = [1 for i in range(length)]

    for i in range(length):
        for j in range(index):
            if input_list[j] < input_list[i]:
                dp[i] = max(dp[i], dp[j]+1)

    res = 0
    for i in range(length):
        if dp[i] > res:
            res = dp[i]

    return res


'''
找零问题
包含找零的策略
coins：1，2，5
amount：11
'''
def coinChange(coins, amount):
    dp = [amount+1 for i in range(amount + 1)]

    dp[0] = 0
    change_list = []

    for i in range(1, amount+1):
        for coin in coins:
            if i - coin < 0:
                continue
            dp[i] = min(dp[i], dp[i-coin] + 1)
    
    index = amount
    while index:
        for coin in coins:
            if dp[index] == dp[index-coin] + 1:
                change_list.append(coin)
                index = index - coin
                break

    return dp[-1], change_list


'''
编辑距离
包含具体修改方式
intention execution 5
horse ros 3
'''
def editdistance(s1, s2):
    length1 = len(s1) + 1
    length2 = len(s2) + 1
    choice = ['same', 'delete', 'add', 'edit']
    edit_list = []

    dp = [[0 for j in range(length2)] for i in range(length1)]

    for i in range(1, length1):
        dp[i][0] = i

    for j in range(1, length2):
        dp[0][j] = j

    for i in range(1, length1):
        for j in range(1, length2):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i][j-1] + 1, dp[i-1][j] + 1, dp[i-1][j-1] + 1)

    i = length1 - 1
    j = length2 - 1
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            edit_list.append('same')
            i = i - 1
            j = j - 1
        else:
            if dp[i][j] == dp[i][j-1] + 1:
                edit_list.append('add')
                j = j - 1
            elif dp[i][j] == dp[i-1][j] + 1:
                edit_list.append('delete')
                i = i - 1
            else:
                edit_list.append('modify')
                i = i - 1
                j = j - 1

    return dp[-1][-1], edit_list


'''
给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
s = "abcabcbb" 3
s = "bbbbb" 1
s = "" 0
'''
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_dict = {}
        
        length = len(s)

        if length < 2:
            return length

        max_len = 0
        left = right = 0

        for index in range(length):
            right = index
            char = s[index]

            if char in char_dict.keys() and char_dict[char] >= left:
                curr_len = right - left
                if curr_len > max_len:
                    max_len = curr_len
                left = char_dict[char] + 1
            
            char_dict[char] = index

        curr_len = right - left + 1
        if curr_len > max_len:
            max_len = curr_len

        return max_len


'''
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/add-two-numbers
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1, l2):
        head = curr = ListNode()
        carry_bit = 0

        while l1 and l2:
            curr.next = ListNode()
            curr = curr.next

            add_val = l1.val + l2.val + carry_bit
            if add_val > 9:
                add_val -= 10
                carry_bit = 1
            else:
                carry_bit = 0

            curr.val = add_val

            l1 = l1.next
            l2 = l2.next

        if not l1:
            while l2:
                if carry_bit == 0:
                    curr.next = l2
                    break

                curr.next = ListNode()
                curr = curr.next

                add_val = l2.val + carry_bit
                if add_val > 9:
                    add_val -= 10
                    carry_bit = 1
                else:
                    carry_bit = 0

                curr.val = add_val

                l2 = l2.next

        if not l2:
            while l1:
                if carry_bit == 0:
                    curr.next = l1
                    break
                
                curr.next = ListNode()
                curr = curr.next

                add_val = l1.val + carry_bit
                if add_val > 9:
                    add_val -= 10
                    carry_bit = 1
                else:
                    carry_bit = 0

                curr.val = add_val

                l1 = l1.next

        if carry_bit == 1:
            curr.next = ListNode()
            curr = curr.next

            curr.val = carry_bit

        return head.next
                

'''
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/two-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''
class Solution:
    def twoSum(self, nums, target):
        temp_dict = dict()

        length = len(nums)

        for index in range(length):
            sub_val = target - nums[index]
            if sub_val in temp_dict.keys() and temp_dict[sub_val] != index:
                left = min(index, temp_dict[sub_val])
                right = max(index, temp_dict[sub_val])
                return [left, right]

            temp_dict[nums[index]] = index

        return []

'''
if __name__ == '__main__':
    # coin change
    print('coin change')
    coins = [1, 2, 5]
    amount = 13
    result, change_list = coinChange(coins, amount)
    print(result)
    print(change_list)
    print()

    # edit distance
    print('edit distance')
    s1 = 'intention'
    s2 = 'execution'
    result, edit = editdistance(s1, s2)
    print(result)
    print(s1)
    print(s2)
    edit.reverse()
    print(edit)
    print()

    # two sum
    print('two sum')
    # nums = [2,7,11,15]
    # target = 9
    nums = [3,2,4]
    target = 6
    test_class = Solution()
    result = test_class.twoSum(nums, target)
    print(result)
'''