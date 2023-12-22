import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torchvision


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class ReshapeTransformTarget:
    def __init__(self, number_classes):
        self.number_classes = number_classes

    def __call__(self, target):
        if torch.is_tensor(target):
            target = target.unsqueeze(0).unsqueeze(1)
        else:
            target = torch.tensor(target).unsqueeze(0).unsqueeze(1)
        target_onehot = torch.zeros((1, self.number_classes))

        return target_onehot.scatter_(1, target.long(), 1).squeeze(0)


class myDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.data = images
        self.targets = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        data = self.data[i, :].numpy()
        target = self.targets[i]
        # data, label = self.data[item].numpy(), self.targets[item]
        data = Image.fromarray(data)

        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            target = self.target_transform(target)

        if self.targets is not None:
            return (data, target)
        else:
            return data

    def __len__(self):
        return (len(self.data))


class YinYangDataset(Dataset):
    def __init__(self, r_small=0.1, r_big=0.5, size=1000, seed=42, transform=None, target_transform=None, sub_class=False):
        super(YinYangDataset, self).__init__()
        # using a numpy RNG to allow compatibility to other deep learning frameworks
        self.rng = np.random.RandomState(seed)
        self.transform = transform
        self.target_transform = target_transform
        self.r_small = r_small
        self.r_big = r_big
        self.vals = []  # private data
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


def generate_N_targets_label(targets, number_per_class, output_neurons):
    multi_targets = list(map(lambda x: np.asarray(range(number_per_class*x, number_per_class*(x+1))), targets))
    mlb = MultiLabelBinarizer(classes=range(output_neurons))
    N_targets = mlb.fit_transform(multi_targets)

    return torch.from_numpy(N_targets)


def Semisupervised_dataset(train_set, targets, output_neurons, n_class, labeled_number, transform, seed=1):

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


def returnMNIST(jparams, validation=False):

    # Define the Transform
    if jparams['convNet']:
        transforms_type = [torchvision.transforms.ToTensor()]
    else:
        transforms_type = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                         # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                         ReshapeTransform((-1,))])

    # Train set
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                            transform=transforms_type,
                                            target_transform=ReshapeTransformTarget(10))


    # Validation set
    if validation:
        (X_train, X_validation,
         Y_train, Y_validation) = train_test_split(train_set.data, train_set.targets,
                                                                     test_size=0.1, random_state=34, stratify=train_set.targets)

        train_set = myDataset(X_train, Y_train, transform=transforms_type, target_transform=ReshapeTransformTarget(10))
        validation_set = myDataset(X_validation, Y_validation, transform=transforms_type, target_transform=None)

    # Class set and Layer set
    if jparams['classLabel_percentage'] == 1:
        class_set = myDataset(train_set.data, train_set.targets, transform=transforms_type, target_transform=None)
        layer_set = train_set
    else:
        class_set = splitClass(train_set.data, train_set.targets, jparams['classLabel_percentage'], seed=34,
                               transform=transforms_type)
        layer_set = splitClass(train_set.data, train_set.targets, jparams['classLabel_percentage'], seed=34,
                               transform=transforms_type,
                               target_transform=ReshapeTransformTarget(10))

    # Supervised set and Unsupervised set
    if jparams['semi_seed'] < 0:
        semi_seed = None
    else:
        semi_seed = jparams['semi_seed']

    supervised_dataset, unsupervised_dataset = Semisupervised_dataset(train_set.data,  train_set.targets,
                                                                      jparams['fcLayers'][-1], jparams['n_class'],
                                                                      jparams['trainLabel_number'],
                                                                      transform=transforms_type, seed=semi_seed)
    # Test set
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms_type)

    # load the dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=jparams['batchSize'], shuffle=True)
    if validation:
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=jparams['test_batchSize'], shuffle=False)
    class_loader = torch.utils.data.DataLoader(class_set, batch_size=jparams['test_batchSize'], shuffle=False)
    layer_loader = torch.utils.data.DataLoader(layer_set, batch_size=jparams['test_batchSize'], shuffle=True)
    supervised_loader = torch.utils.data.DataLoader(supervised_dataset, batch_size=jparams['pre_batchSize'],
                                                    shuffle=True)
    unsupervised_loader = torch.utils.data.DataLoader(unsupervised_dataset, batch_size=jparams['batchSize'],
                                                      shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=jparams['test_batchSize'], shuffle=False)

    if validation:
        return train_loader, validation_loader, class_loader, layer_loader, supervised_loader, unsupervised_loader
    else:
        return train_loader, test_loader, class_loader, layer_loader, supervised_loader, unsupervised_loader


def returnCifar10(jparams, validation=False):
    # Define the Transform
    transform_type = torchvision.transforms.ToTensor()
    if jparams['convNet']:
        train_transform_type = torchvision.transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),  # Rotate by up to 10 degrees
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomGrayscale(p=0.1),
            torchvision.transforms.ToTensor()
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform_type = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ReshapeTransform((-1,))])

    # class name of cifar10
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Train set
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=train_transform_type,
                                             target_transform=ReshapeTransformTarget(10))
    # Validation set
    if validation:
        (X_train, X_validation,
         Y_train, Y_validation) = train_test_split(torch.tensor(train_set.data), torch.tensor(train_set.targets),
                                                                     test_size=0.1, random_state=34, stratify=train_set.targets)

        train_set = myDataset(X_train, Y_train, transform=train_transform_type, target_transform=ReshapeTransformTarget(10))
        validation_set = myDataset(X_validation, Y_validation, transform=transform_type, target_transform=None)

    # Class set and Layer set
    if jparams['classLabel_percentage'] == 1:
        class_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                                 transform=train_transform_type)
        layer_set = train_set
    else:
        class_set = splitClass(torch.tensor(train_set.data), torch.tensor(train_set.targets),
                               jparams['classLabel_percentage'], seed=34,
                               transform=train_transform_type)

        layer_set = splitClass(torch.tensor(train_set.data), torch.tensor(train_set.targets),
                               jparams['classLabel_percentage'], seed=34,
                               transform=train_transform_type,
                               target_transform=ReshapeTransformTarget(10))

    # Supervised set and Unsupervised set
    if jparams['semi_seed'] < 0:
        semi_seed = None
    else:
        semi_seed = jparams['semi_seed']

    supervised_dataset, unsupervised_dataset = Semisupervised_dataset(torch.tensor(train_set.data), torch.tensor(train_set.targets),
                                                                      jparams['fcLayers'][-1], jparams['n_class'],
                                                                      jparams['trainLabel_number'],
                                                                      transform=train_transform_type, seed=semi_seed)
    # Test set
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_type)

    # load the dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=jparams['batchSize'], shuffle=True)
    if validation:
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=jparams['test_batchSize'],
                                                        shuffle=False)
    class_loader = torch.utils.data.DataLoader(class_set, batch_size=jparams['test_batchSize'], shuffle=False)
    layer_loader = torch.utils.data.DataLoader(layer_set, batch_size=jparams['test_batchSize'], shuffle=True)
    supervised_loader = torch.utils.data.DataLoader(supervised_dataset, batch_size=jparams['pre_batchSize'],
                                                    shuffle=True)
    unsupervised_loader = torch.utils.data.DataLoader(unsupervised_dataset, batch_size=jparams['batchSize'],
                                                      shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=jparams['test_batchSize'], shuffle=False)

    if validation:
        return train_loader, validation_loader, class_loader, layer_loader, supervised_loader, unsupervised_loader
    else:
        return train_loader, test_loader, class_loader, layer_loader, supervised_loader, unsupervised_loader
