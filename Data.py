import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class ReshapeTransformTarget:
    def __init__(self, number_classes):
        self.number_classes = number_classes

    def __call__(self, target):
        target = torch.tensor(target).unsqueeze(0).unsqueeze(1)
        target_onehot = torch.zeros((1, self.number_classes))

        return target_onehot.scatter_(1, target.long(), 1).squeeze(0)


class DigitsDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None, target_transforms=None):
        self.x = images
        self.y = labels
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, i):
        data = self.x[i, :]
        target = self.y[i]

        if self.transforms:
            data = self.transforms(data)

        if self.target_transforms:
            target = self.target_transforms(target)

        if self.y is not None:
            return (data, target)
        else:
            return data

    def __len__(self):
        return (len(self.x))


class YinYangDataset(Dataset):
    def __init__(self, r_small=0.1, r_big=0.5, size=1000, seed=42, transform=None, target_transform=None, sub_class=False):
        super(YinYangDataset, self).__init__()
        # using a numpy RNG to allow compatibility to other deep learning frameworks
        self.rng = np.random.RandomState(seed)
        self.transform = transform
        self.target_transform = target_transform
        self.r_small = r_small
        self.r_big = r_big
        self.__vals = [] # private data
        self.__cs = []
        self.class_names = ['yin', 'yang', 'dot']
        for i in range(size):
            # keep num of class instances balanced by using rejection sampling
            # choose class for this sample
            goal_class = self.rng.randint(3)
            x, y, c = self.get_sample(goal=goal_class)
            # add mirrod axis values
            x_flipped = 1. - x
            y_flipped = 1. - y
            val = np.array([x, y, x_flipped, y_flipped])
            self.__vals.append(val)
            self.__cs.append(c)
        if sub_class:
            class_data, remain_data, \
            class_target, remain_target = train_test_split(self.__vals, self.__cs, test_size=0.9,
                                                           random_state=seed, stratify=self.__cs)
            self.__vals = class_data
            self.__cs = class_target

    def get_sample(self, goal=None):
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x,y coordinates
            x, y = self.rng.rand(2) * 2. * self.r_big
            # check if within yin-yang circle
            if np.sqrt((x - self.r_big)**2 + (y - self.r_big)**2) > self.r_big:
                continue
            # check if they have the same class as the goal for this sample
            c = self.which_class(x, y)
            if goal is None or c == goal:
                found_sample_yet = True
                break
        return x, y, c

    def which_class(self, x, y):
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small
        if is_circles:
            return 2
        return int(is_yin)

    def dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big)**2 + (y - self.r_big)**2)

    def dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big)**2 + (y - self.r_big)**2)

    def __getitem__(self, index):
        data = self.__vals[index].copy()
        target = self.__cs[index]
        #sample = (self.__vals[index].copy(), self.__cs[index])
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
            #sample = self.transform(sample)
            # TODO to compare with the MNIST and to do the onehot coding
        sample = (data, target)
        return sample

    def __len__(self):
        return len(self.__cs)


class ClassDataset(Dataset):
    def __init__(self, root, test_set, seed, transform=None, target_transform=None):

        seedfile = 'seed' + str(seed) + '.txt'
        filePath = os.path.join(root, seedfile)
        images_indices = np.loadtxt(filePath).astype(int)
        self.data = test_set.data[images_indices, :]
        self.transform = transform
        self.targets = np.array(test_set.targets)[images_indices]
        self.target_transform = target_transform

    def __getitem__(self, item):
        img, label = self.data[item].numpy(), self.targets[item]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.targets)


class ValidationDataset(Dataset):
    # This dataset is only used for hyperparameter research
    def __init__(self, root, rest_set, seed, transform=None, target_transform=None):
        seedfile = 'validSeed' + str(seed) + '.txt'
        filePath = os.path.join(root, seedfile)
        images_indices = np.loadtxt(filePath).astype(int)
        self.data = rest_set.data[images_indices, :]
        self.transform = transform
        self.targets = rest_set.targets[images_indices]
        self.target_transform = target_transform

    def __getitem__(self, item):
        img, label = self.data[item].numpy(), self.targets[item].numpy()
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.targets)


class HypertestDataset(Dataset):
    # This dataset is only used for hyperparameter research
    def __init__(self, root, rest_set, seed, transform=None, target_transform=None):

        seedfile = 'validSeed' + str(seed) + '.txt'
        filePath = os.path.join(root, seedfile)
        delete_indices = np.loadtxt(filePath).astype(int).tolist()
        total_indices = np.arange(len(rest_set.targets))
        images_indices = np.delete(total_indices, delete_indices)
        self.data = rest_set.data[images_indices, :]
        self.transform = transform
        self.targets = rest_set.targets[images_indices]
        self.target_transform = target_transform

    def __getitem__(self, item):
        img, label = self.data[item].numpy(), self.targets[item].numpy()
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.targets)