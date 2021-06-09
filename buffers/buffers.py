import torch
import numpy as np
import random
from collections import defaultdict, deque


class Random:

    def __init__(self, memory_size, shape):
        self.memory_size = memory_size
        self.memory_used = 0
        self.data = torch.empty([memory_size, *shape])
        self.labels = torch.empty(memory_size, dtype=torch.long)
        self.save_or_not = [True, False]

    def update_memory(self, input, target):
        for j, (x, y) in enumerate(zip(input, target)):
            if self.memory_used < self.memory_size:
                self.data[self.memory_used] = x
                self.labels[self.memory_used] = y
                self.memory_used += 1
            elif random.choice(self.save_or_not):
                random_place = random.randint(0, self.memory_used-1)
                self.data[random_place] = x
                self.labels[random_place] = y

    def sample(self, k=10):
        if self.memory_used == 0:
            return None, None
        if k >= self.memory_used:
            return self.data[:self.memory_used], self.labels[:self.memory_used]
        indices = random.sample(range(self.memory_used), k)
        return self.data[indices], self.labels[indices]


class Reservoir:

    def __init__(self, *, memory_size, shape, batch_size=None):
        self.memory_size = memory_size
        self.memory_used = 0
        self.data = torch.empty([memory_size, *shape])
        self.labels = torch.empty(memory_size, dtype=torch.long)
        self.n = 0  # number of already observed samples from the data stream
        self.batch_size = batch_size

    def update_memory(self, input, target):
        for j, (x, y) in enumerate(zip(input, target)):
            if self.memory_used < self.memory_size:
                self.data[self.memory_used] = x
                self.labels[self.memory_used] = y
                self.memory_used += 1
            else:
                i = random.randint(0, self.n + j - 1)
                if i < self.memory_size:
                    self.data[i] = x
                    self.labels[i] = y
        self.n += len(input)

    def sample(self, k):
        if self.memory_used == 0:
            return None, None
        if self.batch_size:
            k = self.batch_size
        if k >= self.memory_used:
            return self.data[:self.memory_used], self.labels[:self.memory_used]
        indices = random.sample(range(self.memory_used), k)
        return self.data[indices], self.labels[indices]


class CBRS:

    def __init__(self, memory_size, shape):
        self.memory_size = memory_size
        self.memory_used = 0
        self.data = torch.empty([memory_size, *shape])
        self.labels = torch.empty(memory_size, dtype=torch.long)
        self.nc = defaultdict(lambda: 0)  # number of already observed samples of class c from the data stream
        self.class2indices = defaultdict(lambda: [])

    def update_memory(self, input, target):
        for x, y in zip(input, target):
            if self.memory_used < self.memory_size:
                self.data[self.memory_used] = x
                self.labels[self.memory_used] = y
                # print(y)
                self.class2indices[y.item()].append(self.memory_used)
                self.nc[y.item()] += 1
                self.memory_used += 1
            elif not self.__is_full(y):
                l, l_indices = self.__get_the_largest()
                idx = np.random.choice(l_indices)
                self.data[idx] = x
                self.labels[idx] = y
                # print(y)
                self.class2indices[l].remove(idx)
                self.class2indices[y.item()].append(idx)
                self.nc[y.item()] += 1
            else:
                mc = len(self.class2indices[y.item()])
                nc = self.nc[y.item()]  # should i increment nc here also? what is nc?
                u = np.random.uniform()
                if u <= mc / nc:
                    idx = np.random.choice(self.class2indices[y.item()])
                    self.data[idx] = x

    def __is_full(self, class_id):
        """A class is full if it currently is, or has been in one of the previous time steps, the largest class."""
        class_id = class_id.item()
        class_indices = self.class2indices[class_id]
        for c, indices in self.class2indices.items():
            if class_id != c and len(indices) > len(class_indices):
                return False
        return True

    def __get_the_largest(self):
        """When a certain class contains the most instances among all the different classes present in the memory, we will call it the largest.
        Two or more classes can be the largest, if they are equal in size and also the most numerous"""

        the_largest_class = 0
        the_largest_indices = []
        for c, indices in self.class2indices.items():
            if len(indices) > len(the_largest_indices):
                the_largest_class = c
                the_largest_indices = indices
        return the_largest_class, the_largest_indices

    def sample(self, k=10):
        """In our case, we propose the use of a custom replay sampling scheme, where the probability of replaying a certain stored instance is inversely
        proportional to the number of stored instances of the same class."""

        if self.memory_used == 0:
            return None, None

        if k >= self.memory_used:
            return self.data[:self.memory_used], self.labels[:self.memory_used]
        probs = {class_id: len(indices) / self.memory_used for class_id, indices in
                 self.class2indices.items()}  # percentage of class images in memory
        probs = {class_id: 1 - perc if 0 < perc < 1 else perc for class_id, perc in probs.items()}
        classes = [class_id for class_id in range(len(probs))]
        probs = [probs[class_id] for class_id in
                 classes]  # assuming that tasks (classes) are in increasing order 0,1,2,...9
        probs = [prob / sum(probs) for prob in probs]
        # print(probs)
        classes_ids = np.random.choice(classes, size=k, p=probs, replace=True)
        classes_ids = list(classes_ids)
        # print('drawn classes: ', classes_ids)
        indices = []
        for class_id in classes:
            exampels_per_class = classes_ids.count(class_id)
            class_indices = np.random.choice(self.class2indices[class_id], size=exampels_per_class, replace=False)
            class_indices = list(class_indices)
            indices += class_indices
        return self.data[indices], self.labels[indices]

