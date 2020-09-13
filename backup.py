#剑指Offer 32-3 从上到下打印二叉树3
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        queue = collections.deque()
        res = []
        index = 1

        if not root:
            return res
        else:
            queue.append(root)

        while queue:
            temp = []
            for _ in range(len(queue)):
                node = queue.popleft()
                temp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if index%2==1:
                res.append(temp)
            else:
                res.append(temp[::-1])
            index += 1
        
        return res


#剑指Offer 32-2 从上到下打印二叉树2
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# import copy
# class Solution:
#     def levelOrder(self, root: TreeNode) -> List[List[int]]:
#         if not root: return []
#         res, queue = [], collections.deque()
#         queue.append(root)
#         while queue:
#             tmp = []
#             for _ in range(len(queue)):
#                 node = queue.popleft()
#                 tmp.append(node.val)
#                 if node.left: queue.append(node.left)
#                 if node.right: queue.append(node.right)
#             res.append(tmp)
#         return res

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        queue = collections.deque()
        res = []
        index = 0
        arr = []
        if not root:
            return res
        else:
            queue.append(root)
            queue.append(index)
        
        while queue:
            node = queue.popleft()
            temp = queue.popleft()
            if temp==index:
                arr.append(node.val)
            else:
                index += 1
                res.append(copy.deepcopy(arr))
                arr = []
                arr.append(node.val)
            
            if node.left:
                queue.append(node.left)
                queue.append(index+1)
            
            if node.right:
                queue.append(node.right)
                queue.append(index+1)
        
        res.append(arr)
        return res


#剑指Offer 32-1 从上到下打印二叉树
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        #层序遍历
        queue = collections.deque()
        res = []
        if not root:
            return res
        else:
            queue.append(root)

        while queue:
            temp = queue.popleft()
            res.append(temp.val)
            if temp.left:
                queue.append(temp.left)
            if temp.right:
                queue.append(temp.right)
        
        return res


#剑指Offer 31 栈的压入、弹出序列
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        if len(pushed)==0:
            return True

        stack = []
        index = 0
        for val in pushed:
            stack.append(val)
            while stack and stack[-1]==popped[index]:
                stack.pop()
                index += 1
            
        return not stack


#剑指Offer 30 包含min函数的栈
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.minstack = []
        self.stack = []


    def push(self, x: int) -> None:
        self.stack.append(x)
        if not self.minstack:
            self.minstack.append(x)
        else:
            min = self.minstack.pop()
            if x<=min:
                self.minstack.append(min)
                self.minstack.append(x)
            else:
                self.minstack.append(min)


    def pop(self) -> None:
        res = self.stack.pop()
        min = self.minstack.pop()
        if res!=min:
            self.minstack.append(min)



    def top(self) -> int:
        res = self.stack.pop()
        self.stack.append(res)
        return res


    def min(self) -> int:
        res = self.minstack.pop()
        self.minstack.append(res)
        return res



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.min()


#剑指Offer 29 顺时针打印矩阵
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        ri = len(matrix)
        if ri==0:
            return matrix
        rj = len(matrix[0])
        if rj==0:
            return matrix
        li = lj = 0
        res = []
        
        while True:
            for j in range(lj, rj):
                res.append(matrix[li][j])
            li += 1
            if li>=ri:
                break

            for i in range(li, ri):
                res.append(matrix[i][rj-1])
            rj -= 1
            if lj>=rj:
                break

            for j in range(rj-1, lj-1, -1):
                res.append(matrix[ri-1][j])
            ri -= 1
            if li>=ri:
                break
           
            for i in range(ri-1, li-1, -1):
                res.append(matrix[i][lj])
            lj += 1
            if lj>=rj:
                break
                
        return res


#剑指Offer 28 对称二叉树
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        mirror = self.getmirror(root)
        if not root:
            return True
        return self.comparenode(root, mirror)

    def comparenode(self, A, B):
        if not A and not B:
            return True
        if not A and B:
            return False
        if A and not B:
            return False

        if A.val==B.val:
            return self.comparenode(A.left, B.left) and self.comparenode(A.right, B.right)
        else:
            return False

    def getmirror(self, root):
        if not root:
            return None
        temp = TreeNode(root.val)
        temp.left = self.getmirror(root.right)
        temp.right = self.getmirror(root.left)

        return temp


#剑指Offer 27 二叉树的镜像
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        temp = TreeNode(root.val)
        temp.left = self.mirrorTree(root.right)
        temp.right = self.mirrorTree(root.left)
        return temp


#剑指Offer 26 树的子结构
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        def recur(A, B):
            if not B: return True
            if not A or A.val != B.val: return False
            return recur(A.left, B.left) and recur(A.right, B.right)

        return bool(A and B) and (recur(A, B) or self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B))

# class Solution:
#     def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
#         if not B:
#             return False
#         else:
#             return self.dfs(A, B)
        
#     def dfs(self, A, B):
#         if (not A) and B:
#             return False
#         elif not B:
#             return True
        
#         if A.val==B.val:
#             res1 = self.dfs(A.left, B.left)
#             res2 = self.dfs(A.right, B.right)
#             res = res1 and res2
#         else:
#             res1 = self.dfs(A.left, B)
#             res2 = self.dfs(A.right, B)
#             res = res1 or res2

#         return res


#剑指Offer 25 合并两个排序链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        elif not l2:
            return l1

        if l1.val<l2.val:
            temp = l1
            l1 = l1.next
        else:
            temp = l2
            l2 = l2.next
            
        temp.next = self.mergeTwoLists(l1,l2)

        return temp
# class Solution:
#     def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
#         cur = dum = ListNode(0) #使用头结点
#         while l1 and l2:
#             if l1.val < l2.val:
#                 cur.next, l1 = l1, l1.next
#             else:
#                 cur.next, l2 = l2, l2.next
#             cur = cur.next
#         cur.next = l1 if l1 else l2
#         return dum.next


#剑指Offer 24 反转链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        while head:
            temp = head.next
            head.next = pre
            pre = head
            head = temp
        return pre

        # last = None
        # while head:
        #     head.next, last, head = last, head, head.next
        # return last


#剑指Offer 22 链表中倒数第k个结点
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        i = 0
        temp = head
        while temp:
            temp = temp.next
            i += 1
        num = i - k
        for _ in range(num):
            head = head.next
        return head


#剑指Offer 21 调整数组顺序使奇数位于偶数前面
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        lens = len(nums)
        if lens==0:
            return nums
        
        left = 0
        right = lens-1

        while left != right:
            if nums[left]%2==0: #偶数
                nums[left], nums[right] = nums[right], nums[left]
                right -= 1
            else:
                left += 1
        
        return nums


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