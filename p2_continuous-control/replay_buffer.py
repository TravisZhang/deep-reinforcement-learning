import numpy as np
import random
import math
from typing import List
from copy import deepcopy
from collections import namedtuple, deque
import gc
import datetime
import torch
import sys, os
from memory_profiler import profile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Experience = namedtuple("Experience",
                        field_names=[
                            "state", "action", "reward", "next_state", "done",
                            "priority"
                        ])

# Ref: https://nn.labml.ai/rl/dqn/replay_buffer.html

# Tree: sum tree
# key: priority**alpha, value: sum of priority**alpha of self and children


class BinaryNode:

    def __init__(self, key, value, exp=None):
        self.key = key  # priority**alpha
        self.value = value  # sum of priority**alpha of self and children
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.experience = exp
        self.timestamp = str(datetime.datetime.now().timestamp())

    def Reset(self, key):
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.key = key
        self.value = key
        if type(self.experience) == Experience:
            self.experience._replace(priority=self.key)

    def UpdateValue(self):
        self.value = self.key
        if self.left_child is not None:
            self.value += self.left_child.value
        if self.right_child is not None:
            self.value += self.right_child.value

    def __str__(self) -> str:
        output_str = 'key: ' + format(self.key, '.3f') + ' value: ' + format(self.value, '.3f') + ' t: ' + self.timestamp
        if self.parent is not None:
            output_str += ' pt: ' + self.parent.timestamp
            if self.parent.left_child == self:
                output_str += ' left'
            else:
                output_str += ' right'
        # if self.left_child is not None:
        #     output_str += ' lt: ' + self.left_child.timestamp
        # if self.right_child is not None:
        #     output_str += ' rt: ' + self.right_child.timestamp
        return output_str
        pass


class BinaryTree:

    def __init__(self, capacity=int(1e6), alpha=0.5):
        self.root = None
        self.counter = 0

        self.samples = []
        self.weights = []

        self.capacity = capacity
        self.alpha = alpha
        self.max_priority_alpha = 1.0

        self.oldest_node = None

    def Sum(self):
        if self.root is None:
            return 0.0
        return self.root.value

    def Min(self):
        node = self.FindMinimum(self.root)
        if node is None:
            return 0.0
        return node.key

    def Size(self):
        return self.counter

    def AddValue(self, current_node: BinaryNode, value):
        current_node.value += value

    def UpdateValue(self, current_node: BinaryNode):
        if current_node is None:
            return

        current_node.UpdateValue()
        # print('Updating !!! current_node: ', current_node)
        while current_node.parent is not None:
            current_node = current_node.parent
            # if current_node.parent is not None:
            #     print('current_node:', current_node, ' parent: ',
            #           current_node.parent)
            # else:
            #     print('current_node: ', current_node)
            current_node.UpdateValue()

    def InsertNode(self, key=None, input_node: BinaryNode = None, exp=None):
        self.counter += 1
        parent = None
        current_node = self.root
        insert_key = self.max_priority_alpha
        # print('max_priority_alpha:', self.max_priority_alpha)
        if key is not None:
            # print('using existing key !!!!! ', key)
            insert_key = key
        insert_node = BinaryNode(insert_key, insert_key, exp)
        if input_node is not None:
            # print('using existing node !!!!! ', input_node)
            insert_node = input_node
            insert_node.Reset(insert_key)

        if current_node is None:
            self.root = insert_node
            self.oldest_node = insert_node
            return

        while current_node is not None:
            parent = current_node
            self.AddValue(current_node, insert_key)
            # print('current_node:', current_node)
            if insert_key < current_node.key:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child

        insert_node.parent = parent

        if insert_node.key < parent.key:
            parent.left_child = insert_node
        else:
            parent.right_child = insert_node

        # if exceed max capacity, delete root node
        if self.counter > self.capacity:
            self.DeleteNode(self.root)

    def FindMinimum(self, node: BinaryNode) -> BinaryNode:
        current_node = node
        while current_node.left_child is not None:
            current_node = current_node.left_child
        return current_node

    def FindMaximum(self, node: BinaryNode) -> BinaryNode:
        current_node = node
        while current_node.right_child is not None:
            current_node = current_node.right_child
        return current_node

    def Search(self, key):
        current_node = self.root
        while current_node is not None and current_node.key != key:
            if key < current_node.key:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        return current_node

    def Transplant(self, node0: BinaryNode, node1: BinaryNode):
        if node0.parent is None:
            self.root = node1
            node1.parent = None
        elif node0 == node0.parent.left_child:
            node0.parent.left_child = node1
        else:
            node0.parent.right_child = node1
        if node1 is not None:
            node1.parent = node0.parent

    def DeleteNode(self, target_node: BinaryNode):
        if target_node is None:
            print('Delete node is None !!!')
            return

        self.counter -= 1
        self.counter = max(self.counter, 0)
        if target_node.left_child is None:
            # print('step 00')
            self.Transplant(target_node, target_node.right_child)
            # print('target_node: ', target_node, ' parent: ',
            #       target_node.parent)
            self.UpdateValue(target_node.parent)
        elif target_node.right_child is None:
            # print('step 01')
            self.Transplant(target_node, target_node.left_child)
            # print('target_node: ', target_node, ' parent: ',
            #       target_node.parent)
            self.UpdateValue(target_node.parent)
        else:
            # print('target_node: ', target_node, ' left_child: ',
            #       target_node.left_child, ' right_child: ',
            #       target_node.right_child)
            # print('target left left child: ',
            #       target_node.left_child.left_child)
            # print('find pred')
            predecessor = self.FindMinimum(target_node.right_child)
            # print('predecessor before: ', predecessor, ' parent: ',
            #       predecessor.parent)
            if predecessor.parent != target_node:
                # print('step 1')
                self.Transplant(predecessor, predecessor.right_child)
                self.UpdateValue(predecessor.parent)
                # print('predecessor trans 1: ', predecessor)
                # print('right_child: ', predecessor.right_child)
                predecessor.right_child = target_node.right_child
                predecessor.right_child.parent = predecessor
                # print('predecessor change right: ', predecessor)
                # print('right_child: ', predecessor.right_child)
            # print('step 2')
            # print('target_node: ', target_node, ' parent: ',
            #       target_node.parent)
            self.Transplant(target_node, predecessor)
            # print('predecessor trans 2: ', predecessor)
            # print('left_child: ', predecessor.left_child)
            predecessor.left_child = target_node.left_child
            predecessor.left_child.parent = predecessor
            # print('predecessor change left: ', predecessor)
            # print('left_child: ', predecessor.left_child)
            if predecessor == predecessor.parent:
                print('error for loop found !!!')
                input()
            self.UpdateValue(predecessor)
        # print(gc.get_count())
        del target_node
        # gc.collect()
        # print(gc.collect())
        # print(gc.get_count())
        # print('target_node deleted:',target_node)

    def FindNodeForSumUnder(self, input_sum):
        current_node = self.root
        if current_node is None or input_sum > current_node.value:
            return current_node

        parent = None
        while current_node is not None:
            parent = current_node
            left_right_sum = current_node.value - current_node.key
            if input_sum >= left_right_sum:
                break
            left_child_value = 0.0
            if current_node.left_child is not None:
                left_child_value = current_node.left_child.value
            if input_sum < left_child_value:
                current_node = current_node.left_child
            else:
                input_sum -= left_child_value
                current_node = current_node.right_child

        return parent
        pass

    def Sample(self, batch_size, beta):
        self.samples = []
        self.weights = []
        if self.root is None or self.Size() == 0:
            print('Sample failed !!!, tree empty')
            return self.samples, self.weights

        # sample a node
        for i in range(batch_size):
            p = random.random() * self.Sum()
            node = self.FindNodeForSumUnder(p)
            self.samples.append(node)

        # calculate weight
        min_prob = self.Min() / self.Sum()
        one_over_max_weight = (self.Size() * min_prob)**beta
        for sample in self.samples:
            prob = sample.key / self.Sum()
            weight = (self.Size() * prob)**(-beta)
            self.weights.append(weight * one_over_max_weight)

        experiences = []
        for sample in self.samples:
            experiences.append(sample.experience)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences
                       if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences
                       if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences
                       if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones), torch.tensor(self.weights)
        pass

    def UpdatePriorities(self, priorities, save_path=None):
        if len(self.samples) == 0:
            return
        if len(self.samples) != len(priorities):
            print('priority len not equal: ', len(priorities))
            return

        for sample, priority in zip(self.samples, priorities):
            # if sample.parent is not None:
            #     print('\nupdate sample: ', sample, ' parent: ', sample.parent,
            #           ' priority: ', priority)
            # else:
            #     print('\nupdate sample: ', sample, ' priority: ', priority)

            priority_alpha = priority**self.alpha
            self.max_priority_alpha = max(self.max_priority_alpha,
                                          priority_alpha)
            # if type(sample.experience) is Experience:
            #     # print('replacing experience p: ', sample.experience.priority, ' with p: ', priority)
            #     sample.experience._replace(priority=priority)
            # print('\ndeleting')
            self.DeleteNode(sample)
            # self.Print()
            # print('\ninserting')
            self.InsertNode(priority_alpha, sample)
            # self.Print()
            # if save_path is not None:
            #     input_str = 'delete and insert: ' + sample.timestamp
            #     self.Print(save_path, input_str)
        pass

    def InOrderTreeWalk(self, node: BinaryNode):
        if node is not None:
            # print('self before: ', node)
            self.InOrderTreeWalk(node.left_child)
            if node.parent is not None:
                print('self: ', node, ' parent: ', node.parent)
            else:
                print('self: ', node)
            self.sum_value += node.key
            self.InOrderTreeWalk(node.right_child)
        pass

    def Print(self, save_path=None, input_str=None):
        if save_path is not None:
            sys.stdout = open(save_path, 'a')
        if input_str is not None:
            print('input_str: ', input_str)
        print('!!!!! print tree, sum: ', self.Sum(), ' size: ', self.Size())
        self.sum_value = 0
        self.InOrderTreeWalk(self.root)
        print('sum_value: ', self.sum_value)
        if save_path is not None:
            sys.stdout.close()


# debug log & problem locate:
# 1. first the alpha=1 version works but alpha=0.5 not(there is always a loop in tree)
# 2. the reason is because alpha=1 sample each node only once but alpha=0.5 sample one node multiple times
# 3. and before in UpdatePriorities when the node is deleted, we insert completely new node, so the old node is lost
# 4. so for the next same node to be updated, there will be error
# solution:
# each time we insert, we insert exactly the node we deleted, so that the next same node will be found still in the tree
# also we are using timestamp to label the nodes

if __name__ == "__main__":
    key_list = [10, 5, 6, 3, 7, 4, 8, 9, 3, 2, 11, 13]
    tree = BinaryTree()
    for key in key_list:
        tree.InsertNode(key**tree.alpha)

    # create save_path
    cur_folder = os.getcwd()
    file_name = 'log.txt'
    save_path = cur_folder + '/logs/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    tree.Print()
    print('sum of list:', sum(key_list))

    current_node = tree.Search(3**tree.alpha)
    print('\nfound current node:', current_node)
    print('deleting node')
    tree.DeleteNode(current_node)
    tree.Print()

    current_node = tree.Search(10**tree.alpha)
    print('\nfound current node:', current_node)
    print('deleting node')
    tree.DeleteNode(current_node)
    tree.Print()

    current_node = tree.Search(5**tree.alpha)
    print('\nfound current node:', current_node)
    print('deleting node')
    tree.DeleteNode(current_node)
    tree.Print()
    print('current_node after del:', current_node)

    print('\nsampling tree')
    input_sum_list = [65, 40, 38, 32, 25, 16, 8, 4, 1]
    input_node_list = []
    for input_sum in input_sum_list:
        current_node = tree.FindNodeForSumUnder(input_sum)
        print('found current node:', current_node, ' for sum: ', input_sum)
        input_node_list.append(current_node)

    tree.samples = input_node_list
    random_priorities = []
    for sample in tree.samples:
        random_priorities.append((int(random.random() * 10) + 1))

    print('\nupdating priorities, sum: ',
          sum(np.array(random_priorities)**tree.alpha))
    tree.UpdatePriorities(random_priorities)
    print('done')
    tree.Print()

    print([p**tree.alpha for p in random_priorities])
    print('\nrandom priorities and samples:')
    for sample, p in zip(tree.samples, random_priorities):
        print('priority: ', p**tree.alpha, ' sample: ', sample)

    print('\nsampling new tree')
    sample_num = int(tree.Sum()) * 2
    sample_delta = tree.Sum() / sample_num
    for i in range(sample_num + 2):
        input_sum = i * sample_delta
        current_node = tree.FindNodeForSumUnder(input_sum)
        print('found current node:', current_node, ' for sum: ', input_sum)

    # delete minimum & maximum
    current_node = tree.FindMinimum(tree.root)
    print('\nfound minimum current node:', current_node)
    print('deleting node')
    tree.DeleteNode(current_node)
    tree.Print()

    current_node = tree.FindMaximum(tree.root)
    print('\nfound maximum current node:', current_node)
    print('deleting node')
    tree.DeleteNode(current_node)
    tree.Print()

    # tree.Print(save_path+file_name)
    
    # pressure test
    total_capacity = int(1e6)
    random_priorities = []
    for i in range(total_capacity):
        random_priorities.append(random.random())
    current_time = datetime.datetime.now().timestamp()
    max_priority = max(random_priorities)
    sum_priority = sum(random_priorities)
    delta_time = datetime.datetime.now().timestamp()-current_time
    print('\ndelta time: ', delta_time, ' max priority: ', max_priority)
    