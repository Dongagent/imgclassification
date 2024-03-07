# datasets.py
import torch, random
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

# set random seed
def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

torch_fix_seed()

# set my root dir 
root_dir = 'archive/train/'
train_split = 0.9
batch_size = 64

# define the training transforms and augmentations
train_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# # control group, not using augmentations
# train_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

valid_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# initial entire training datasets
full_dataset = datasets.ImageFolder(root_dir, transform=train_transform)

# define my dataset class
class ImbalancedScrewDataset(Dataset):
    def __init__(self, data, classes, transform=None, mode='train'):
        
        self.data = data
        self.transform = transform
        self.classes = classes
        # compute pairs and labels
        self.img = [i[0] for i in self.data]
        self.labels = [i[1] for i in self.data]
        self.mode = mode
        
    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im = Image.open(self.img[idx])
        im = im.convert('RGB')

        # Apply transformations if any
        if self.transform:
            im = self.transform(im)

        return im, self.labels[idx]
    
raw_data = full_dataset.samples
classes = full_dataset.classes

# Create train and val datasets by indexing the ImbalancedScrewDataset instance
from sklearn.model_selection import train_test_split
raw_data
X_split = [i[0] for i in raw_data]
Y_split = [i[1] for i in raw_data]

X_train, X_val, y_train, y_val = train_test_split(X_split, Y_split, test_size=0.2, stratify=Y_split)
train_dataset = sorted(list(zip(X_train, y_train)))
val_dataset = sorted(list(zip(X_val, y_val)))

# verify the composition
# nclasses = len(classes)
# count = [0] * nclasses
# for item in train_dataset:
#     count[item[1]] += 1
# print(count)
# nclasses = len(classes)
# count = [0] * nclasses
# for item in val_dataset:
#     count[item[1]] += 1
# print(count)


train_dataset = ImbalancedScrewDataset(train_dataset, classes=classes, transform=train_transform, mode='train')
val_dataset = ImbalancedScrewDataset(val_dataset, classes=classes, transform=valid_transform, mode='valid')


# rebalance training datasets
def make_weights_for_balanced_classes(images, nclasses):                     
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = 1. / float(count[i])
    # print(f"weights per class: {weight_per_class}")
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# get weights for training dataset
weights = make_weights_for_balanced_classes(train_dataset.data, len(train_dataset.classes))

# sampler
samples_weight = torch.from_numpy(np.array(weights))
samples_weigth = samples_weight.double()
sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

# training loader using sampler, while using sampler, shuffle should be False
train_loader = DataLoader(
    train_dataset, batch_size = batch_size, sampler = sampler, num_workers = 4, shuffle = False
)
# not using sampler
# train_loader = DataLoader(
#     full_dataset, batch_size = batch_size, shuffle = True, num_workers = 4
# )

valid_loader = DataLoader(
    val_dataset, batch_size = batch_size, shuffle = False, num_workers = 4
)

