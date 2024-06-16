import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_accuracy(loss_list, accuracy_list,model_name):
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
    plt.plot(range(1, epochs + 1), accuracy_list, label=model_name, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

import matplotlib.pyplot as plt

def plot_accuracy_comparison(accuracy_lists, model_names):
    epochs = len(accuracy_lists[0])
    
    # Plotting accuracy
    plt.figure(figsize=(10, 5))
    
    # Define colors for different models
    colors = ['green', 'orange', 'blue', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    for i, accuracy_list in enumerate(accuracy_lists):
        plt.plot(range(1, epochs + 1), accuracy_list, label=model_names[i], color=colors[i % len(colors)])
        
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()





def plot_feature_maps(model, x, img,device='cpu'):
    x.to(device)

    logits, att_mats = model(x.unsqueeze(0))
    
    # Convert the attention matrices to the device of the input tensor
    att_mats = [att_mat.to(device) for att_mat in att_mats]
    
    fig, axes = plt.subplots(ncols=len(att_mats) + 1, figsize=(5 * (len(att_mats) + 1), 5))

    # Plot the original image
    axes[0].set_title("Original")
    axes[0].imshow(img)

    # Plot the attention maps for each layer
    for i, att_mat in enumerate(att_mats, 1):
        # Average the attention weights across all heads
        att_mat = torch.mean(att_mat.squeeze(1), dim=1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(att_mat.size(), device=device)
        joint_attentions[0] = att_mat[0]

        for n in range(1, att_mat.size(0)):
            joint_attentions[n] = torch.matmul(att_mat[n], joint_attentions[n - 1])

        # Attention from the output token to the input space
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")

        # Plot the attention map
        axes[i].set_title(f"Layer {i}")
        axes[i].imshow(result)

    plt.show()

def plot_feature_maps_strided(model, x, img, device='cpu'):
    x.to(device)

    logits, att_mats = model(x.unsqueeze(0))
    
    # Convert the attention matrices to the device of the input tensor
    att_mats = [att_mat.to(device) for att_mat in att_mats]
    
    fig, axes = plt.subplots(ncols=len(att_mats) + 1, figsize=(5 * (len(att_mats) + 1), 5))

    # Plot the original image
    axes[0].set_title("Original")
    axes[0].imshow(img)

    # Plot the attention maps for each layer
    for i, att_mat in enumerate(att_mats, 1):
        # Average the attention weights across all heads
        att_mat = torch.mean(att_mat.squeeze(1), dim=1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(att_mat.size(), device=device)
        joint_attentions[0] = att_mat[0]

        for n in range(1, att_mat.size(0)):
            joint_attentions[n] = torch.matmul(att_mat[n], joint_attentions[n - 1])

        # Attention from the output token to the input space
        v = joint_attentions[-1]
        # grid_size = int(np.sqrt(att_mat.size(-1)))
        # mask = v[0,1:].reshape(grid_size, grid_size).detach().cpu().numpy()
        grid_size = int(np.sqrt(v.size(0)))  # Calculate grid size based on the size of the tensor
        mask = v.reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")

        # Plot the attention map
        axes[i].set_title(f"Layer {i}")
        axes[i].imshow(result)

    plt.show()
