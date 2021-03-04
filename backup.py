'''
双指针：
    滑动窗口： 76 567 438 3 *** 双指针
    nsum： 1 15 18 *** 排序+双指针/哈希表
    移除元素： 27 ***
    链表： 142 206 ***

区间问题： 1288 56 986 435 452 ***

二叉树： 124 99
        297 1908 ***
        226 114 116 105 ***
        654 105 106 652 ***

二叉搜索树：230 538 1038 ***

单调栈：496 503 ***
单调队列：239 ***

翻转链表： 25 234 ***

贪心算法： 435 452 55 45

动态规划：
	最大子数组： 53
	最长公共子序列： 1143 583 712
	背包问题： 416 518 0-1背包
	股票买卖：
    编辑距离： 72
    信封嵌套： 354
    最长递增子序列： 300

回溯： N皇后

BFS： 111 752

哈希表： 1 454 

'''
# 25. K 个一组翻转链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        # 翻转[a,b)之间的链表
        def reverse(a, b):
            pre = None
            curr = nxt = a
            while curr is not b:
                nxt = curr.next
                curr.next = pre
                pre = curr
                curr = nxt

            return pre
        
        if head==None:
            return head

        a = b = head

        # 如果有k个链表则翻转，否则直接返回
        for i in range(k):
            if b==None:
                return head
            b = b.next
        
        newHead = reverse(a, b)
        a.next = self.reverseKGroup(b, k)

        return newHead
        
# 452. 用最少数量的箭引爆气球
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:

        list_len = len(points)
        if list_len == 0:
            return 0

        def sort_rule(point):
            return point[1]
        
        points.sort(key=sort_rule)
        res = 1
        curr_end = points[0][1]

        for index in range(1, list_len):
            if curr_end<points[index][0]:
                res += 1
                curr_end = points[index][1]
            
        return res

# 435. 无重叠区间
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # 排序+贪心

        list_len = len(intervals)
        if list_len == 0:
            return 0

        def sort_rule(interval):
            return interval[1]
        
        intervals.sort(key=sort_rule)
        res = 1
        curr_end = intervals[0][1]

        for index in range(1, list_len):
            if curr_end<=intervals[index][0]:
                res += 1
                curr_end = intervals[index][1]
            
        return list_len - res
            

###### 排序+画图 start ############
# 986. 区间列表的交集
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        i= j = 0
        res = []
        while i<len(firstList) and j<len(secondList):
            listA = firstList[i]
            listB = secondList[j]

            # 不相交
            if listA[0]>listB[1]:
                j+=1
            elif listA[1]<listB[0]:
                i+=1
            # 相交
            else:
                minBound = max(listA[0], listB[0])
                maxBound = min(listA[1], listB[1])
                res.append([minBound, maxBound])

                if listA[1]<listB[1]:
                    i+=1
                else:
                    j+=1
        
        return res

# 1288. 删除被覆盖区间
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:

        # 使用起点升序排列，起点相同时使用终点降序排列
        def sort_rule(interval):
            return interval[0], -interval[1]
        
        intervals.sort(key=sort_rule)
        res = []

        for interval in intervals:
            if not res or res[-1][1]<interval[0]:
                res.append(interval)
            elif interval[1]>res[-1][1]:
                res.append(interval)

        return len(res)


# 56. 合并区间
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        def sort_rule(interval):
            return interval[0]
        
        intervals.sort(key=sort_rule)

        res = []

        for interval in intervals:
            if not res or res[-1][1]<interval[0]:
                res.append(interval)
            else:
                res[-1][1] = max(interval[1], res[-1][1])
            
        return res
###### 排序+画图 end ############

###### 链表+指针 start ############
# 206. 反转链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        
        pre = None
        curr = head

        while curr is not None:
            nextN = curr.next
            curr.next = pre
            pre = curr
            curr = nextN
        
        return pre
        

# 142. 环形链表 II
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        slow = fast = head
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                break
        
        if fast is None or fast.next is None:
            return None
        
        slow = head
        while slow is not fast:
            slow = slow.next
            fast = fast.next
        
        return slow


# 27. 移除元素
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:

        list_len = len(nums)
        left = 0
        right = list_len - 1
        
        while right>-1 and nums[right]==val:
            right -= 1

        while left<right:
            if nums[left]==val:
                nums[left], nums[right] = nums[right], nums[left]
                
                while right>-1 and nums[right]==val:
                    right -= 1
            else:
                left += 1
            
        if right<0:
            return 0
        else:
            return left+1
###### 链表+指针 end ############

###### nsum start ############
# 1. 两数之和
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 暴力
        # length = len(nums)
        # for i in range(length-1):
        #     for j in range(i+1, length):
        #         if target == nums[i] + nums[j]:
        #             return [i, j]
        
        # return []

        # 哈希表
        hashtable = {}
        for i in range(len(nums)):
            sub = target - nums[i]
            if sub in hashtable.keys():
                return [hashtable[sub], i]
            hashtable[nums[i]] = i
        
        return []

        # 双指针
        # def twoSum(self, nums, target):
        # list_len = len(nums)
        # left = 0
        # right = list_len - 1
        # res = []
        # while left < right:
        #     temp = nums[left]+nums[right]
        #     if  temp == target:
        #         res.append([nums[left], nums[right]])
        #         while left<right and nums[left]==nums[left+1]:
        #             left += 1
        #         left += 1
        #         while left<right and nums[right]==nums[right-1]:
        #             right -= 1
        #         right -= 1
        #     elif temp < target:
        #         while left<right and nums[left]==nums[left+1]:
        #             left += 1
        #         left += 1
        #     else:
        #         while left<right and nums[right]==nums[right-1]:
        #             right -= 1
        #         right -= 1

        # return res

# 15. 三数之和
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:

        threeSumRes = []
        nums.sort()
        list_len = len(nums)
        index = 0
        while index < list_len-2:
            twoSumRes = self.twoSum(nums[index+1:], -nums[index])
            for val in twoSumRes:
                val.append(nums[index])
                threeSumRes.append(val)
            
            while index<list_len-3 and nums[index]==nums[index+1]:
                index += 1
            index += 1

        return threeSumRes

    def twoSum(self, nums, target):
        list_len = len(nums)
        left = 0
        right = list_len - 1
        res = []
        while left < right:
            temp = nums[left]+nums[right]
            if  temp == target:
                res.append([nums[left], nums[right]])
                while left<right and nums[left]==nums[left+1]:
                    left += 1
                left += 1
                while left<right and nums[right]==nums[right-1]:
                    right -= 1
                right -= 1
            elif temp < target:
                while left<right and nums[left]==nums[left+1]:
                    left += 1
                left += 1
            else:
                while left<right and nums[right]==nums[right-1]:
                    right -= 1
                right -= 1

        return res

# 18. 四数之和
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        fourSumRes = []
        nums.sort()
        list_len = len(nums)
        index = 0
        while index < list_len-3:
            threeSumRes = self.threeSum(nums[index+1:], target-nums[index])
            for val in threeSumRes:
                val.append(nums[index])
                fourSumRes.append(val)
            
            while index<list_len-4 and nums[index]==nums[index+1]:
                index +=1
            
            index += 1
        
        return fourSumRes

    def threeSum(self, nums: List[int], target:int) -> List[List[int]]:
        threeSumRes = []
        # nums.sort()
        list_len = len(nums)
        index = 0
        while index < list_len-2:
            twoSumRes = self.twoSum(nums[index+1:], target-nums[index])
            for val in twoSumRes:
                val.append(nums[index])
                threeSumRes.append(val)
            
            while index<list_len-3 and nums[index]==nums[index+1]:
                index += 1
            index += 1

        return threeSumRes

    def twoSum(self, nums, target):
        list_len = len(nums)
        left = 0
        right = list_len - 1
        res = []
        while left < right:
            temp = nums[left]+nums[right]
            if  temp == target:
                res.append([nums[left], nums[right]])
                while left<right and nums[left]==nums[left+1]:
                    left += 1
                left += 1
                while left<right and nums[right]==nums[right-1]:
                    right -= 1
                right -= 1
            elif temp < target:
                while left<right and nums[left]==nums[left+1]:
                    left += 1
                left += 1
            else:
                while left<right and nums[right]==nums[right-1]:
                    right -= 1
                right -= 1

        return res
###### nsum end ############
        
###### 滑动窗口 start ############
# 438. 找到字符串中所有字母异位词
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        need = {}
        for char in p:
            need[char] = need.setdefault(char, 0) + 1
        
        window = {}
        left = right = valid = 0
        res = []

        while right < len(s):
            curr_char = s[right]
            right += 1

            if curr_char in need.keys():
                window[curr_char] = window.setdefault(curr_char, 0) + 1
                if window[curr_char] == need[curr_char]:
                    valid += 1
            
            while right-left == len(p):
                if valid == len(need):
                    res.append(left)
                
                del_char = s[left]
                left += 1
                
                if del_char in need.keys():
                    if window[del_char] == need[del_char]:
                        valid -= 1
                    window[del_char] -= 1

        return res 


# 567. 字符串的排列
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        need = {}
        for char in s1:
            need[char] = need.setdefault(char, 0) + 1
        
        window = {}
        left = right = valid= 0

        while right < len(s2):
            curr_char = s2[right]
            right += 1

            if curr_char in need.keys():
                window[curr_char] = window.setdefault(curr_char, 0) + 1
                if window[curr_char] == need[curr_char]:
                    valid += 1
            
            while right-left == len(s1):
                if valid == len(need):
                    return True
                
                del_char = s2[left]
                left += 1

                if del_char in need.keys():
                    if window[del_char] == need[del_char]:
                        valid -= 1
                    window[del_char] -= 1
            
        return False
            

# 76. 最小覆盖子串
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = {}
        for char in t:
            need[char] = need.setdefault(char, 0) + 1
        
        window = {}

        left = right = valid = 0
        length = float('inf') 
        res = ''

        while right < len(s):
            curr_char = s[right]
            right += 1

            if curr_char in need.keys():
                window[curr_char] = window.setdefault(curr_char, 0) + 1
                if window[curr_char] == need[curr_char]:
                    valid += 1
            
            while valid == len(need.keys()):
                if right-left < length:
                    length = right - left
                    res = s[left:right]
                
                del_char = s[left]
                left += 1

                if del_char in need.keys():
                    if window[del_char] == need[del_char]:
                        valid -= 1
                    window[del_char] -= 1
            
        return res


# 3. 无重复字符的最长子串
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 滑动窗口
        length = len(s)
        left = right = res = 0

        window = dict()

        while right < length:
            curr_char = s[right]
            right += 1

            window[curr_char] =  window.setdefault(curr_char, 0) + 1

            while window[curr_char]>1:
                del_char = s[left]
                left += 1
                window[del_char] -= 1

            res = max(res, right-left)

        return res
###### 滑动窗口 end ############


# 剑指 Offer 42. 连续子数组的最大和
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
    	'''
    	动态规划算法
    	'''
        dp = [0]*len(nums)

        dp[0] = nums[0]

        for i in range(1, len(nums)):
            dp[i] = nums[i] + max(dp[i-1], 0)
        
        return max(dp)



# 剑指 Offer 68 - II. 二叉树的最近公共祖先
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:

        if not root or root==p or root==q:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if not left: 
            return right
        if not right:
            return left
        return root


# 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        def get_target_node(root, target_node):
            path = []
            path.append(root)
            while target_node != root:
                
                if target_node.val > root.val:
                    root = root.right
                    path.append(root)
                else:
                    root = root.left
                    path.append(root)
        
            return path

        # root_p = root_q = root
        path_p = get_target_node(root, p)
        path_q = get_target_node(root, q)
        res = None

        for u, v in zip(path_p, path_q):
            if u == v:
                res = u
            else:
                break
                
        return res



# 剑指 Offer 66. 构建乘积数组
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        a_len = len(a)

        if not a_len:
            return []

        # 一般解法
        # b, left, right = [0]*a_len, [0]*a_len, [0]*a_len
        # left[0] = right[a_len-1] = 1

        # for i in range(1, a_len):
        #     left[i]  = left[i-1] * a[i-1]

        # for i in range(a_len-2, -1, -1):
        #     right[i] = right[i+1] * a[i+1]

        # for i in range(a_len):
        #     b[i] = left[i] * right[i]

        # return b

        #节省空间
        b, tmp = [0]*a_len, 1
        b[0] = 1

        for i in range(1,a_len):
            b[i] = b[i-1] * a[i-1]

        for i in range(a_len-2, -1, -1):
            tmp *= a[i+1]
            b[i] *= tmp

        return b


# 剑指 Offer 65. 不用加减乘除做加法
class Solution:
    def add(self, a: int, b: int) -> int:

        x = 0xffffffff
        a, b = a&x, b&x
        while b !=0:
            a, b = (a^b), (a&b)<<1 &x
        
        return a if a<=0x7fffffff else ~(a^x)


# 剑指 Offer 64. 求1+2+…+n
class Solution:
    def __init__(self):
        self.res = 0

    def sumNums(self, n: int) -> int:
        n > 0 and self.sumNums(n-1)
        self.res += n 
        return self.res 


# 剑指 Offer 63. 股票的最大利润
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        记录历史最低价，并考虑当天价格与历史最低价的差值
        '''
        max_profit = 0
        min_price = math.inf

        for price in prices:
            max_profit = max(price-min_price, max_profit)
            min_price = min(price, min_price)

        return max_profit


# 剑指 Offer 62. 圆圈中最后剩下的数字 
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        '''
        约瑟夫问题
        f(n,m) = [f(n-1, m) + m] % n , n>1
                 0, n=1 
        ''' 

        f = 0
        for i in range(2, n+1):
            f = (m+f) % i
        
        return f


# 剑指 Offer 61. 扑克牌中的顺子
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        
        nums.sort()
        min_val = -1
        max_val = nums[-1]
        for i in range(len(nums)-1):
            if nums[i] != 0 and nums[i]==nums[i+1]:
                return False
        
        for val in nums:
            if val != 0:
                min_val = val
                break

        if max_val-min_val<5:
            return True
        else:
            return False


# 剑指 Offer 60. n个骰子的点数
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        temp_list = [[0 for _ in range(6*n+1)] for _ in range(n+1)]

        for i in range(7):
            temp_list[1][i] = 1

        for i in range(2,n+1):
            for j in range(i, 6*i+1):
                for cur in range(1,7):
                    if j>=cur+1:
                        temp_list[i][j] += temp_list[i-1][j-cur]

        sum_value = 6**n
        res = []
        for i in range(n, 6*n+1):
            res.append(temp_list[n][i]*1.0 / sum_value)

        return res


# 剑指 Offer 59 - II. 队列的最大值
class MaxQueue:

    def __init__(self):
        self.max_queue = collections.deque()
        self.queue = collections.deque()


    def max_value(self) -> int:
        if self.max_queue:
            return self.max_queue[0]
        else:
            return -1


    def push_back(self, value: int) -> None:
        self.queue.append(value)
    
        while self.max_queue and self.max_queue[-1]<value:
            self.max_queue.pop()
        self.max_queue.append(value)

    def pop_front(self) -> int:
        if self.queue:
            pop_value = self.queue.popleft()
            if pop_value==self.max_queue[0]:
                self.max_queue.popleft()
            return pop_value
        else:
            return -1
# Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(value)
# param_3 = obj.pop_front()

# 超时
class MaxQueue:

    def __init__(self):
        self.curr_max = -1
        self.queue = collections.deque()

    def max_value(self) -> int:
        if self.curr_max == -1 and self.queue:
            return max(self.queue)
        else:
            return self.curr_max


    def push_back(self, value: int) -> None:
        self.queue.append(value)
        if value>self.curr_max:
            self.curr_max = value


    def pop_front(self) -> int:
        if self.queue:
            pop_value = self.queue.popleft()
            if pop_value==self.curr_max:
                self.curr_max = -1
            return pop_value
        else:
            return -1

    def get_curr_max(self):
        if not self.queue:
            self.curr_max == -1
        else:
            self.curr_max = max(self.queue)


# 剑指 Offer 59 - I. 滑动窗口的最大值
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 处理空队列
        if len(nums)==0:
            return []

        i,j = 0,k-1
        #在窗口k中获得最大值及其下标
        def get_curr_max(i, j):
            curr_max = nums[i]
            curr_max_index = i
            for index in range(i, j+1):
                if nums[index] > curr_max:
                    curr_max = nums[index]
                    curr_max_index = index
            return curr_max, curr_max_index
        #获得初始时的下标
        curr_max, curr_max_index = get_curr_max(i,j)
        res = [curr_max]
        #滑动窗口 n-k+1次
        while j<len(nums)-1:
            i += 1
            j += 1
            #如果上一轮的最大值在窗口外，则重新获取最大值
            if curr_max_index<i:
                curr_max, curr_max_index = get_curr_max(i,j)
                res.append(curr_max)
            #如果上一轮最大值还在窗口内
            #如果新加入的值更大，则修改当前最大值
            elif nums[j]>curr_max:
                    curr_max_index = j
                    curr_max = nums[j]
                    res.append(curr_max)
            #否则维持原状
            else:
                res.append(curr_max)
        
        return res


# 剑指 Offer 58 - II. 左旋转字符串
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:

        sub_str = s[:n]
        result = s[n:]

        return result + sub_str


# 剑指 Offer 58 - I. 翻转单词顺序
class Solution:
    def reverseWords(self, s: str) -> str:

        str_list = s.strip().split(' ')
        res = ''
        for val in str_list[::-1]:
            if val != '':
                res += val
                res += ' '
        
        return res.strip()


# 剑指 Offer 57 - II. 和为s的连续正数序列
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:

        if target<3:
            return []
        res = []    
        left, right = 1,2
        while left < right and (right-1 <= target/2):
            sum = (left+right)*(right-left+1)/2
            if sum == target:
                res.append([x for x in range(left, right+1)])
                right += 1
            elif sum>target:
                left += 1
            else:
                right += 1

        return res


# 剑指 Offer 57. 和为s的两个数字
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 方法一 使用双重循环，利用数组有序提前终止
        # stop_index = len(nums)
        # for index in range(len(nums)):
        #     if nums[index] >= target:
        #         stop_index = index
        #         break
        
        # for i in range(stop_index-1, -1, -1):
        #     for j in range(stop_index):
        #         if i!=j:
        #             if nums[i] + nums[j] == target:
        #                 return [nums[j], nums[i]]
        #             elif nums[i] + nums[j] > target:
        #                 break

        # 使用双指针，时间复杂度0(n)
        i, j = 0, len(nums)-1
        while i<j:
            s = nums[i] + nums[j]
            if s>target:
                j -= 1
            elif s== target:
                return [nums[i], nums[j]]
            else:
                i += 1
        
        return []


# 剑指 Offer 56 - II. 数组中数字出现的次数 II
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        #方法1 使用字典求解
        # counter_dic = collections.Counter(nums)

        # for key, val in counter_dic.items():
        #     if val == 1:
        #         return key
        
        #方法2 使用位运算求解
        binary = [0]*32

        for num in nums:
            for i in range(32):
                binary[i] += (num & 1)
                num >>= 1
        res = 0
        for i in range(32):
            if binary[i] % 3 != 0:
                res += pow(2, i)

        return res


# 剑指 Offer 56 - I. 数组中数字出现的次数
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        #方法1 字典
        # counter_dic = {}

        # for num in nums:
        #     counter_dic[num] = counter_dic.get(num, 0) + 1

        # # counter_dic = collections.Counter(nums)

        # result = []

        # for key, val in counter_dic.items():
        #     if val == 1:
        #         result.append(key)
        

        # return result

        #方法二 位运算
        xor_result = functools.reduce(lambda x,y: x^y, nums)

        a = b = 0

        div = 1
        while div & xor_result == 0:
            div <<= 1
        
        for num in nums:
            if num & div:
                a ^= num
            else:
                b ^= num
        
        return [a,b]       


# 剑指 Offer 55 - II. 平衡二叉树
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if root == None:
            return True
        elif abs(self.get_deepth(root.left) - self.get_deepth(root.right)) > 1:
            return False
        else:
            return self.isBalanced(root.left) and self.isBalanced(root.right)
        

    def get_deepth(self, root):
        if root == None:
            return 0
        elif root.left==None and root.right==None:
            return 1
        else:
            return max(self.get_deepth(root.left), self.get_deepth(root.right)) + 1


# 剑指 Offer 55 - I. 二叉树的深度
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def get_max_deepth(self, root):
        if root == None:
            return 0
        elif root.left == None and root.right == None:
            return 1
        else: 
            return max(self.get_max_deepth(root.left), self.get_max_deepth(root.right)) + 1

    def maxDepth(self, root: TreeNode) -> int:
        return self.get_max_deepth(root)


# 剑指 Offer 54. 二叉搜索树的第k大节点
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:

        def traversal(root):
            if not root:
                return 
            
            traversal(root.right)
            if self.k == 0:
                return
            self.k -= 1
            if self.k == 0:
                self.res = root.val
            traversal(root.left)

        self.k = k
        # self.res = -100
        traversal(root)

        return self.res


# 剑指 Offer 53 - II. 0～n-1中缺失的数字
class Solution:
    def missingNumber(self, nums: List[int]) -> int:

        left = 0
        right = len(nums) - 1
        # mid = 0 

        # if left == right:
        #     if left == nums[left]:
        #         return left + 1
        #     else:
        #         return left

        while left < right:
            mid = (right + left) // 2
            if mid == nums[mid]:
                left = mid + 1
            else:
                right = mid
                
        if left == len(nums)-1 and left == nums[-1]:
            # if left == nums[-1]:
            return left + 1
        
        # if right == 0:
        #     return 0
        
        return left


# 剑指 Offer 53 - I. 在排序数组中查找数字 I
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        length = len(nums)
        index = 0
        count = 0

        while index < length:
            if target < nums[index]:
                return count
            elif target == nums[index]:
                count += 1
                index += 1
            else:
                index += 1
            
        return count 


# 剑指 Offer 52. 两个链表的第一个公共节点
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        
        lenA = 0
        lenB = 0

        tempA = headA
        tempB = headB
        while tempA:
            lenA += 1 
            tempA = tempA.next
        
        while tempB:
            lenB += 1
            tempB = tempB.next
        
        subAB = lenA - lenB

        if subAB > 0:
            while subAB:
                headA = headA.next
                subAB -= 1
        elif subAB < 0:
            while subAB:
                headB = headB.next
                subAB += 1
        
        while headA :
            if headA == headB:
                return headA
            headA = headA.next
            headB = headB.next
        
        return None


#剑指 Offer 38 字符串的排列
class Solution:
    def permutation(self, s: str) -> List[str]:
        def dfs(inputlist, string ,result):
            if len(inputlist)==0:
                temp = copy.deepcopy(string)
                result.append(temp)
                return 
            
            for j in range(len(inputlist)):
                string += inputlist[j]
                templist = copy.deepcopy(inputlist)
                templist.pop(j)
                dfs(templist, string, result)
                string = string[:-1]

        strlist = [item for item in s]

        result = []
        string = ''
        dfs(strlist, string, result)
        return list(set(result))


#剑指 Offer 36 二叉搜索树与双向链表
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root:
            return root

        def inordertrave(root):
            if not root:
                return None
            
            inordertrave(root.left)
            nodelist.append(root)
            inordertrave(root.right)

        nodelist = []
        inordertrave(root)
        listsize = len(nodelist)
        temp = listsize - 1

        for index in range(listsize):
            nodelist[index].right = nodelist[(index+1)%listsize]
            nodelist[index].left = nodelist[(index+temp)%listsize]

        return nodelist[0]


#剑指 Offer 35 复杂链表的复制
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        def dfs(head):
            if not head:
                return None
            if head in visited.keys():
                return visited[head]

            temp = Node(head.val)
            visited[head] = temp
            temp.next = dfs(head.next)
            temp.random = dfs(head.random)

            return temp
        
        visited = {}
        return dfs(head)


#剑指Offer 40 最小的K的数
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        arr.sort()
        return arr[:k]


#剑指Offer 39 数组中出现次数超过一半的数字
#可使用hashmap
#摩尔投票法
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums)//2]


#剑指Offer 34 二叉树中和为某一值的路径
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        if not root:
            return []
        result = []
        cursum = 0
        temp = []

        self.dfs(root, cursum, sum, temp, result)
        return result

    def dfs(self, root, cursum, sum, temp, result):
        
        temp.append(root.val)
        cursum += root.val
        if not root.left and not root.right:
            if cursum==sum:
                result.append(copy.deepcopy(temp))
                temp.pop()
                cursum -= root.val
                return
            else:
                temp.pop()
                cursum -= root.val
                return

        if root.left:
            self.dfs(root.left, cursum, sum, temp, result)
            
        if root.right:
            self.dfs(root.right, cursum, sum, temp, result)  
        temp.pop()


#剑指Offer 33 二叉搜索树的后序遍历序列
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        if len(postorder)<=1:
            return True
        
        root = postorder[-1]
        lefttree = []
        righttree = []
        k = -1
        for i in range(len(postorder)-1):
            if postorder[i]<root:
                lefttree.append(postorder[i])
            else:
                k = i
                break
        if k != -1:
            for i in range(k, len(postorder)-1):
                if postorder[i]<root:
                    return False
                else:
                    righttree.append(postorder[i])
        
        return self.verifyPostorder(lefttree) and self.verifyPostorder(righttree)


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