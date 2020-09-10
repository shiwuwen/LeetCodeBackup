#剑指Offer 18 删除链表结点
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        pre = head

        if pre.val==val:
            head = pre.next
            return head

        while pre.next:
            if pre.next.val == val:
                temp = pre.next
                pre.next = temp.next
                return head
            
            pre = pre.next


#剑指Offer 17 打印1到最大的n位数
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        #未考虑n大于int32的情况
        nums = 10**n
        result = []
        for i in range(1, nums):
            result.append(i)
        
        return result

        # def dfs(x):
        #     if x==n:
        #         res = ''.join(nums)
        #         result.append(int(res))
        #         return
            
        #     for i in range(10):
        #         nums.append(str(i))
        #         dfs(x+1)
        #         nums.pop()
        
        # nums = []
        # result = []
        # dfs(0)
        # return result


#剑指Offer 16 数值的整数次方
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x==0:
            return 0
        
        res = 1
        if n<0:
            x, n = 1/x, -n
        while n:
            if n&1:
                res *= x
            x *= x
            n >>= 1
        return res


#剑指Offer 15 二进制中1的个数
class Solution:
    def hammingWeight(self, n: int) -> int:
        #method1
        #n&1确定最右一位是否为1
        # result = 0
        # while n:
        #     result += n&1
        #     n >>=1
        # return result

        #method2
        #n&(n-1)可消除n中的一个1
        result = 0
        while n:
            n &= n-1
            result += 1
        return result


#剑指Offer 14-1 14-2 剪绳子
class Solution:
    def cuttingRope(self, n: int) -> int:
        #由算术均值不等式得当各段相等时乘积最大
        #由导数得n为e时值最大

        if n<=3:
            return n-1
        
        a,b = n//3, n%3
        if b==0:
            return 3**a
        if b==1:
            return (3**(a-1))*4
        if b==2:
            return (3**a)*2


#剑指Offer 12 矩阵中的路径
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(i, j, k):
            if not 0<=i<len(board) or not 0<=j<len(board[0]) or board[i][j]!=word[k]:
                return False
            
            if k==len(word)-1:
                return True
            
            temp, board[i][j] = board[i][j], -1
            res = dfs(i+1, j, k+1) or dfs(i-1, j, k+1) or dfs(i, j+1, k+1) or dfs(i, j-1, k+1)
            board[i][j] = temp
            return res

        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0):
                    return True
        return False


#剑指Offer 11 旋转数组的最小数字
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        #使用类二分查找来实现
        low = 0
        high = len(numbers)-1

        while low < high:
            mid = (low + high) // 2
            if numbers[mid]>numbers[high]:
                low = mid + 1
            elif numbers[mid]<numbers[high]:
                high = mid #二分查找为 high=mid-1
            else:
                high -= 1
        return numbers[low]


#剑指Offer 10-2 青蛙跳台阶问题
class Solution:
    def numWays(self, n: int) -> int:
        # tatal(n) = total(n-1)+total(n-2)
        # 与斐波那契数列类似
        a = 1 #n-2
        b = 1 #n-1
        for _ in range(n):
            a, b = b, a+b
        
        return a % 1000000007


#剑指Offer 10-1 求斐波那契（Fibonacci）数列的第 n 项
class Solution:
    def fib(self, n: int) -> int:
        # 递归超时
        # if n == 0:
        #     return 0
        # elif n ==1:
        #     return 1
        # else:
        #     result = self.fib(n-1)+ self.fib(n-2)
        #     return  result % 1000000007

        a = 0
        b = 1

        for _ in range(n):
            a, b = b, a+b
        return a % 1000000007


#剑指Offer 09 用两个栈实现队列
# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
class CQueue:

    def __init__(self):
        self.headstack = [-1 for i in range(10001)]
        self.tailstack = [-1 for i in range(10001)]
        self.headtop = 0
        self.tailtop = 0

    def appendTail(self, value: int) -> None:
        if self.tailtop == 0:
            self.headstack[self.headtop] = value
            self.headtop +=1
            return None

        for i in range(self.tailtop-1, -1, -1):
            self.headstack[self.headtop] = self.tailstack[i]
            self.headtop += 1
        self.headstack[self.headtop] = value
        self.headtop += 1
        self.tailtop = 0
        return None

    def deleteHead(self) -> int:
        if self.headtop == 0:
            if self.tailtop == 0:
                return -1
            else:
                self.tailtop -= 1
                return self.tailstack[self.tailtop]

        for i in range(self.headtop-1, -1, -1):
            self.tailstack[self.tailtop] = self.headstack[i]
            self.tailtop += 1
        self.headtop = 0
        self.tailtop -= 1
        return self.tailstack[self.tailtop]


#剑指Offer 07 重建二叉树
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if len(preorder)==0:
            return None
        elif len(preorder)==1:
            return TreeNode(preorder[0])
        else:
            root = TreeNode(preorder[0])
            k = -1
            for i in range(len(inorder)):
                if inorder[i]==preorder[0]:
                    k = i
                    break
            leftinorder = inorder[:k]
            rightinorder = inorder[k+1:]

            leftpreorder = preorder[1:k+1]
            rightpreorder = preorder[k+1:]

            root.left = self.buildTree(leftpreorder, leftinorder)
            root.right = self.buildTree(rightpreorder, rightinorder)
            return root


#剑指Offer 06 从头到尾打印链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        if head is None:
            return []
        
        temp = []
        result = []
        temp.append(head.val)

        while head.next:
            head = head.next
            temp.append(head.val)

        for i in range(len(temp)-1, -1, -1):
            result.append(temp[i])

        return result


#剑指Offer 05 替换空格
class Solution:
    def replaceSpace(self, s: str) -> str:
        result = ''
        if s is None:
            return None
            
        for i in s:
            if i is not ' ':
                result +=i
            else:
                result += '%20'
        
        return result