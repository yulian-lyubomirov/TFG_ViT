import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import sys
from sklearn.cluster import KMeans
import torch
import torch.quantization as quantization
import cv2


def compute_pruning_ammount(epoch):
    # if epoch == 50:
    #     return 0.15

    # elif epoch == 50:
    #     return 0.15

    # elif 95 < epoch < 99:
    #     return 0.01
    if epoch == 30:
        return 0.95

    return 0


def quantize_model(model, test_loader, device, save_path=None):
    # Convert the model to evaluation mode
    model.eval()
    model.to("cpu")
    # Apply quantization to the model
    model = quantization.QuantWrapper(model)
    model.qconfig = quantization.get_default_qconfig("fbgemm", 0)  # Specify version 0
    quantization.prepare(model, inplace=True)
    quantization.convert(model, inplace=True)
    model.to(device)
    # Evaluate the quantized model
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Save the quantized model if a path is provided
    if save_path:
        torch.save(model.state_dict(), save_path)

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def print_gpu_memory_usage():
    print("GPU Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")


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



def kmeans_clustering(X_query, num_clusters):
    B, N, _ = X_query.size()
    X_query_flattened = X_query.view(B * N, -1).detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_indices = kmeans.fit_predict(X_query_flattened)
    cluster_indices = torch.tensor(cluster_indices, device=X_query.device)
    return cluster_indices


def topk_operation(X_query, cluster_indices, num_clusters, k):
    B, N, _ = X_query.size()
    XK = torch.zeros(B, num_clusters, k, X_query.size(-1), device=X_query.device)
    XV = torch.zeros(B, num_clusters, k, X_query.size(-1), device=X_query.device)

    for b in range(B):
        for j in range(num_clusters):
            mask = cluster_indices[b] == j
            relevant_keys = X_query[b][mask]
            topk_indices = torch.topk(
                torch.norm(relevant_keys - X_query[b][j], dim=-1), k
            ).indices
            topk_indices = topk_indices[
                topk_indices < len(relevant_keys)
            ]  # Ensure indices are within bounds
            relevant_keys = relevant_keys[topk_indices]  # Correct shape for indexing
            XK[b][j][: len(relevant_keys)] = relevant_keys
            XV[b][j][: len(relevant_keys)] = X_query[b][mask][topk_indices]
    return XK, XV
