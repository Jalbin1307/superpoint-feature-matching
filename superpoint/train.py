from calendar import EPOCH
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms
import os
from model import SP_Classifier
from tqdm import tqdm
from loss import SPLoss
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEVICE = "cuda" if torch.cuda.is_available else "cpu"
# DEVICE = "cpu"
EPOCHS = 100
criterion = torch.nn.MSELoss()

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    train_loss = 0.0
    ra = 0
    running_accuracy = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)

        loss = criterion(torch.max(out, 1), y)
        print(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        _, predicted = torch.max(out, 1)
        running_accuracy += (predicted == y).sum().item()
        print(running_accuracy)

        total += y.size(0)
        # update progress bar
        loop.set_postfix(loss=loss.item())
    ra += running_accuracy

    # print("ra : {}, n : {}, accuracy : {}%".format(ra, len(train_loader)*8, ra*100/(len(train_loader)*8)))

def main():

    model = SP_Classifier().to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0
    )
    trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.Resize((240, 320)),
                            transforms.Normalize(([0.5]), ([0.5]))
                            ])
    trainset = torchvision.datasets.ImageFolder(root="./Dataset", transform = trans)

    classes = trainset.classes

    train_loader = DataLoader(trainset,
                            batch_size=8,
                            shuffle =True,
                            num_workers=0)

    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, criterion)

    torch.save(model.state_dict(), "my_checkpoint.pth.tar")
if __name__ == "__main__":
    main()