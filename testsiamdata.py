from siamdata import OmniTest
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

if __name__ == '__main__':

    testset=OmniTest("./images_background",transform=transforms.ToTensor())
    trainloader=DataLoader(testset, batch_size=128, num_workers = 4)

    for batch in trainloader:
        print(batch[0][0].shape)
        tensor=batch[0][0]
        image_array = tensor.squeeze().numpy()
        plt.imshow(image_array, cmap='gray')
        plt.axis('off')
        plt.show()
