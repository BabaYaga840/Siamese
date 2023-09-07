import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image

class OmniTest(Dataset):
    def __init__(self,dataPath,transform=None):
        super().__init__()
        self.data={}
        self.num_classes=0
        self.transform=transform
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                self.data[self.num_classes] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    self.data[self.num_classes].append(Image.open(filePath).convert('L'))
                self.num_classes += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = None
        image1 = None
        image2 = None
        if idx % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.data[idx1])
            image2 = random.choice(self.data[idx1])
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.data[idx1])
            image2 = random.choice(self.data[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))

if __name__ == '__main__':
    omnitrain = OmniTest("./images_background")
    print(omnitrain)
    print(omnitrain.__getitem__(0))