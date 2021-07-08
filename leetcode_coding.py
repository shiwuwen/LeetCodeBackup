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
