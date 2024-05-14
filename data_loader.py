import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torchvision.transforms as transforms
import random

def get_data_loader(batch_size, num_workers, data_path, download=False):
    transforms_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transforms_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = datasets.CIFAR100(
        data_path, train=True, download=download, transform=transforms_train
    )
    test_dataset = datasets.CIFAR100(
        root=data_path, train=False, download=download, transform=transforms_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def load_checkpoint(model, path):
    state_dict = torch.load(path, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    # own_state_dict = model.state_dict()
    # for param,name in state_dict.items():
    #     if name in own_state_dict and "head" not in name:
    #         own_state_dict[name].copy_(param)

    torch.cuda.empty_cache()
    # return max_accuracy


def save_checkpoint(epoch, model, optimizer, lr_scheduler, accuracy, path):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "accuracy": accuracy,
        "epoch": epoch,
    }

    save_path = os.path.join(path, f"checkpoint_epoch_{epoch}.pt")
    torch.save(save_state, save_path)


def save_best_model(model, optimizer, lr_scheduler, accuracy, path):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "accuracy": accuracy,
    }

    save_path = os.path.join(path, f"best_model.pt")
    torch.save(save_state, save_path)

def save_lists_to_file(loss_list, accuracy_list, save_path):
    loss_array = np.array(loss_list)
    accuracy_array = np.array(accuracy_list)

    data_array = np.column_stack((loss_array, accuracy_array))

    np.savetxt(
        save_path, data_array, delimiter=",", header="Loss,Accuracy", comments=""
    )

def load_lists_from_file(file_path):
    # Load data from file
    data_array = np.loadtxt(file_path, delimiter=",", skiprows=1)
    
    # Extract loss and accuracy arrays
    loss_list = data_array[:, 0]
    accuracy_list = data_array[:, 1]

    return loss_list, accuracy_list

def get_random_image(img_class,dataset='cifar-100'):
    _,test_loader = get_data_loader(
    1, 2, f"datasets/{dataset}/{dataset}python", download=True
    )
    transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.ToTensor(),
    ]
    )
    random_num = random.randint(0,99)
    img = Image.open(f"datasets/cifar-100/cifar-100-python/cifar-100-png/test/{img_class}/00{random_num}.png")
    x =transform(img)
    return x,img
