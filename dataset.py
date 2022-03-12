import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import args
from utils.augment import distort, random_erasing, blur
from utils.char_utils import get_char_dict, char2id


class SynthHWCL(Dataset):
    def __init__(self, data_dir, label_files, transform=None):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.images = list()
        self.labels = list()
        self.transform = transform
        self.char_dict = get_char_dict(args.char_dict_path)

        for label_file in label_files:
            if os.path.exists(label_file):
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if len(line.split('\t')) == 2:
                            image_path, label = line.split('\t')
                            image_path = os.path.join(self.data_dir, image_path)
                            label_ids = [char2id(char, self.char_dict) for char in label]
                            self.images.append(image_path)
                            self.labels.append(label_ids)

    def __getitem__(self, index):
        image = Image.open(self.images[index])  # Pillow Gray Image
        if self.transform is not None:
            image = self.transform(image)
        image = np.array(image).reshape((1, args.img_H, args.img_W))
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        label = self.labels[index]  # list
        label = torch.IntTensor(label)
        label_length = torch.IntTensor([len(label)])

        return image, label, label_length

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.cat(labels, 0)
    label_lengths = torch.cat(label_lengths, 0)

    return images, labels, label_lengths


transformer = transforms.RandomApply(
    [transforms.ColorJitter(brightness=0.25, contrast=0.25),
     transforms.RandomChoice(
         [transforms.Lambda(lambda img: distort(img, 8, img_type='PIL_Image')),
          transforms.Lambda(lambda img: random_erasing(img, 20, 10)),
          transforms.Lambda(lambda img: blur(img, radius=1)),
          # transforms.Lambda(lambda img: reverse(img))
          ]
     )
     ],
    p=1
)
