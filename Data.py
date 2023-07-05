import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


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


class myDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.x = images
        self.y = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        data = self.x[i, :].numpy()
        target = self.y[i]
        # data, label = self.data[item].numpy(), self.targets[item]
        data = Image.fromarray(data)

        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            target = self.target_transform(target)

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
        self.vals = [] # private data
        self.cs = []
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
            self.vals.append(val)
            self.cs.append(c)
        if sub_class:
            class_data, remain_data, \
            class_target, remain_target = train_test_split(self.vals, self.cs, test_size=0.9,
                                                           random_state=seed, stratify=self.cs)
            self.vals = class_data
            self.cs = class_target

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
        data = self.vals[index].copy()
        target = self.cs[index]
        #sample = (self.vals[index].copy(), self.cs[index])
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
            #sample = self.transform(sample)
            # TODO to compare with the MNIST and to do the onehot coding
        sample = (data, target)
        return sample

    def __len__(self):
        return len(self.cs)


class splitClass(Dataset):
    def __init__(self, x, y, split_ratio, seed, transform=None, target_transform=None):

        class_set_data, rest_data, \
        class_set_targets, rest_targets = train_test_split(x, y, train_size=split_ratio, random_state=seed, stratify=y)

        del(rest_data, rest_targets)

        self.data = class_set_data
        self.transform = transform
        self.targets = class_set_targets
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


# class ClassDataset(Dataset):
#     def __init__(self, root, test_set, seed, transform=None, target_transform=None):
#
#         seedfile = 'seed' + str(seed) + '.txt'
#         filePath = os.path.join(root, seedfile)
#         images_indices = np.loadtxt(filePath).astype(int)
#         self.data = test_set.data[images_indices, :]
#         self.transform = transform
#         self.targets = np.array(test_set.targets)[images_indices]
#         self.target_transform = target_transform
#
#     def __getitem__(self, item):
#         img, label = self.data[item].numpy(), self.targets[item]
#         img = Image.fromarray(img)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             label = self.target_transform(label)
#         return img, label
#
#     def __len__(self):
#         return len(self.targets)


def generate_N_targets_label(targets, number_per_class, output_neurons):
    multi_targets = list(map(lambda x: np.asarray(range(number_per_class*x, number_per_class*(x+1))), targets))
    mlb = MultiLabelBinarizer(classes=range(output_neurons))
    N_targets = mlb.fit_transform(multi_targets)

    return torch.from_numpy(N_targets)


def Semisupervised_dataset(train_set, targets, output_neurons, n_class, labeled_number, transform, seed=None):

    fraction = labeled_number/len(targets)
    # we split the dataset for supervised training and unsupervised training
    X_super, X_unsuper, Y_super, Y_unsuper = train_test_split(train_set, targets, test_size=1 - fraction,
                                                              train_size=fraction, random_state=seed,
                                                              stratify=targets)
    number_per_class = int(output_neurons/n_class)

    # we define the target of supervised learning considering the number of output neurons
    if number_per_class > 1:
        N_Y_super = generate_N_targets_label(Y_super, number_per_class, output_neurons)
    else:
        N_Y_super = torch.nn.functional.one_hot(Y_super, num_classes=-1)

    # we load the target
    dataset_super = myDataset(X_super, N_Y_super, transform=transform, target_transform=None)
    dataset_unsuper = myDataset(X_unsuper, Y_unsuper, transform=transform, target_transform=None) # no one-hot coding
    # dataset_super = torch.utils.data.TensorDataset(transform(X_super), N_Y_super)
    # dataset_unsuper = torch.utils.data.TensorDataset(transform(X_unsuper), Y_unsuper) # no one-hot coding

    return dataset_super, dataset_unsuper


# class ValidationDataset(Dataset):
#     # This dataset is only used for hyperparameter research
#     def __init__(self, root, rest_set, seed, transform=None, target_transform=None):
#         seedfile = 'validSeed' + str(seed) + '.txt'
#         filePath = os.path.join(root, seedfile)
#         images_indices = np.loadtxt(filePath).astype(int)
#         self.data = rest_set.data[images_indices, :]
#         self.transform = transform
#         self.targets = rest_set.targets[images_indices]
#         self.target_transform = target_transform
#
#     def __getitem__(self, item):
#         img, label = self.data[item].numpy(), self.targets[item].numpy()
#         img = Image.fromarray(img)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             label = self.target_transform(label)
#         return img, label
#
#     def __len__(self):
#         return len(self.targets)
#
#
# class HypertestDataset(Dataset):
#     # This dataset is only used for hyperparameter research
#     def __init__(self, root, rest_set, seed, transform=None, target_transform=None):
#
#         seedfile = 'validSeed' + str(seed) + '.txt'
#         filePath = os.path.join(root, seedfile)
#         delete_indices = np.loadtxt(filePath).astype(int).tolist()
#         total_indices = np.arange(len(rest_set.targets))
#         images_indices = np.delete(total_indices, delete_indices)
#         self.data = rest_set.data[images_indices, :]
#         self.transform = transform
#         self.targets = rest_set.targets[images_indices]
#         self.target_transform = target_transform
#
#     def __getitem__(self, item):
#         img, label = self.data[item].numpy(), self.targets[item].numpy()
#         img = Image.fromarray(img)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             label = self.target_transform(label)
#         return img, label
#
#     def __len__(self):
#         return len(self.targets)