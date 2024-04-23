import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.optim as optim


def get_data_loader(batch_size, num_workers, data_path, download=False):
    transforms_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            # transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transforms_test = transforms.Compose(
        [
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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

    own_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state_dict and "head" not in name:
            own_state_dict[name].copy_(param)

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


def print_gpu_memory_usage():
    print("GPU Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")


def save_lists_to_file(loss_list, accuracy_list, save_path):
    loss_array = np.array(loss_list)
    accuracy_array = np.array(accuracy_list)

    data_array = np.column_stack((loss_array, accuracy_array))

    np.savetxt(
        save_path, data_array, delimiter=",", header="Loss,Accuracy", comments=""
    )


def plot_loss_accuracy(loss_list, accuracy_list):
    epochs = len(loss_list)

    # Plotting loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_list, label="Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), accuracy_list, label="Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model, learning_rate):
    has_decay = []  # Parameters with weight decay
    no_decay = []  # Parameters without weight decay
    for name, param in model.named_parameters():
        if "bias" in name:
            # Exclude biases from weight decay
            no_decay.append(param)
        else:
            has_decay.append(param)

    # Define optimizer parameters
    parameters = [
        {"params": has_decay, "weight_decay": 0.01},  # Apply weight decay
        {"params": no_decay, "weight_decay": 0.0},  # No weight decay
    ]

    # Instantiate the optimizer
    optimizer = optim.Adam(parameters, lr=learning_rate)
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model, learning_rate):
    has_decay = []  # Parameters with weight decay
    no_decay = []  # Parameters without weight decay
    for name, param in model.named_parameters():
        if "bias" in name:
            # Exclude biases from weight decay
            no_decay.append(param)
        else:
            has_decay.append(param)

    # Define optimizer parameters
    parameters = [
        {"params": has_decay, "weight_decay": 0.01},  # Apply weight decay
        {"params": no_decay, "weight_decay": 0.0},  # No weight decay
    ]

    # Instantiate the optimizer
    optimizer = optim.Adam(parameters, lr=learning_rate)
    return optimizer
