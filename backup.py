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