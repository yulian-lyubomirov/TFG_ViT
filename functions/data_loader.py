import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
import functions.helpers as helpers
import pandas as pd
from torch import nn


def get_data_loader(batch_size, num_workers, data_path, download=False):
    """
    Creates and returns data loaders for training and testing datasets.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        data_path (str): Path to the data directory.
        download (bool): Whether to download the data if it does not exist at data_path.

    Returns:
        tuple: A tuple containing the training data loader and testing data loader.
    """
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
    """
    Loads a model checkpoint from the given path.

    Args:
        model (torch.nn.Module): The model to load the state dictionary into.
        path (str): Path to the checkpoint file.

    Returns:
        None
    """
    state_dict = torch.load(path, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    torch.cuda.empty_cache()

def save_checkpoint(epoch, model, optimizer, lr_scheduler, accuracy, path):
    """
    Saves the model checkpoint.

    Args:
        epoch (int): The current epoch number.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler used for training.
        accuracy (float): The accuracy of the model.
        path (str): The directory where the checkpoint will be saved.

    Returns:
        None
    """
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
    """
    Saves the best model based on accuracy.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler used for training.
        accuracy (float): The accuracy of the model.
        path (str): The directory where the model will be saved.

    Returns:
        None
    """
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "accuracy": accuracy,
    }

    save_path = os.path.join(path, f"best_model.pt")
    torch.save(save_state, save_path)

def save_lists_to_file(loss_list, accuracy_list, save_path):
    """
    Saves loss and accuracy lists to a CSV file.

    Args:
        loss_list (list): List of loss values.
        accuracy_list (list): List of accuracy values.
        save_path (str): Path to save the CSV file.

    Returns:
        None
    """
    loss_array = np.array(loss_list)
    accuracy_array = np.array(accuracy_list)

    data_array = np.column_stack((loss_array, accuracy_array))

    np.savetxt(
        save_path, data_array, delimiter=",", header="Loss,Accuracy", comments=""
    )

def load_lists_from_file(file_path):
    """
    Loads loss and accuracy lists from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing the loss list and accuracy list.
    """
    data_array = np.loadtxt(file_path, delimiter=",", skiprows=1)
    loss_list = data_array[:, 0]
    accuracy_list = data_array[:, 1]

    return loss_list, accuracy_list

def get_random_image(img_class, dataset='cifar-100'):
    """
    Gets a random image of a specified class from the dataset.

    Args:
        img_class (str): The class of the image to retrieve.
        dataset (str): The dataset to use (default is 'cifar-100').

    Returns:
        tuple: A tuple containing the transformed image tensor and the original image.
    """
    _, test_loader = get_data_loader(1, 2, f"datasets/{dataset}/{dataset}python", download=True)
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
        ]
    )
    random_num = random.randint(0, 99)
    img = Image.open(f"datasets/cifar-100/cifar-100-python/cifar-100-png/test/{img_class}/00{random_num}.png")
    x = transform(img)
    return x, img


def create_comparison_table(models, models_accuracy):
    """
    Creates a comparison table for model performance.

    Args:
        models (list of torch.nn.Module): List of models to compare.
        models_accuracy (list of float): List of accuracies corresponding to the models.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the comparison table.
    """
    models_info = []
    
    for model, accuracy in zip(models, models_accuracy):
        num_params = sum(p.numel() for p in model.parameters())
        depth = helpers.calculate_model_depth(model)
        num_heads = helpers.get_num_heads(model)
        num_cnn_layers = helpers.count_cnn_layers(model)
        
        models_info.append({
            'model_name': model.__class__.__name__,
            'num_params': num_params,
            'accuracy': accuracy,
            'depth': depth,         
            'num_heads': num_heads, 
            'num_cnn_layers': num_cnn_layers,

        })
    
    # Create a DataFrame from the models_info list
    df = pd.DataFrame(models_info)

    # Sort the DataFrame by accuracy in descending order
    df = df.sort_values(by='accuracy', ascending=False)

    return df