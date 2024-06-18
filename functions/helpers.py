import torch
import torch.optim as optim
from sklearn.cluster import KMeans
import torch
import torch.quantization as quantization
import cv2
import torch.nn.functional as F
from torch import nn


def feature_distillation_loss(student_features, teacher_features):
    """
    Compute the feature distillation loss.
    
    Args:
        student_features (list of torch.Tensor): List of feature tensors from the student model.
        teacher_features (list of torch.Tensor): List of feature tensors from the teacher model.
    
    Returns:
        torch.Tensor: The computed feature distillation loss.
    """
    loss = 0.0
    num_layers = len(student_features)
    
    for sf, tf in zip(student_features, teacher_features):
        
        if sf.shape != tf.shape:
            raise ValueError("The shapes of student and teacher feature maps must match.")
        student_features = F.normalize(sf, p=2, dim=1)
        teacher_features = F.normalize(tf, p=2, dim=1)
        
        # Compute MSE loss
        loss += F.mse_loss(student_features, teacher_features)
    
    return loss
    
    # Return the mean loss
    return loss #/ num_layers


def feature_distillation_loss2(student_features, teacher_features):
    """
    Compute the feature distillation loss with knowledge distillation (KD) on specific layers.

    Args:
        student_features (list of torch.Tensor): List of feature tensors from the student model.
        teacher_features (list of torch.Tensor): List of feature tensors from the teacher model.

    Returns:
        torch.Tensor: The computed feature distillation loss with KD.
    """
    loss = 0.0
    
    kd_indices = [0,-1,1]

    for idx in kd_indices:
        sf = student_features[idx]
        tf = teacher_features[idx]

        if sf.shape != tf.shape:
            raise ValueError("The shapes of student and teacher feature maps must match.")
        loss += F.mse_loss(sf, tf)

    return loss / len(kd_indices['student'])  # Return the mean loss

def knowledge_distillation_loss(student_logits, teacher_logits, temperature):
    """
    Compute the knowledge distillation loss using soft targets.
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss

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

def calculate_model_depth(model):
    """
    Calculates the depth of the model.

    Args:
        model (torch.nn.Module): The model whose depth is to be calculated.

    Returns:
        int: The depth of the model.
    """
    # For ViT, the depth can be directly extracted from the transformer attribute
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        return len(model.transformer.layers)
    else:
        def module_depth(module, current_depth=0):
            if not list(module.children()):  # If no child modules, it's a leaf
                return current_depth
            return max(module_depth(child, current_depth + 1) for child in module.children())
        
        return module_depth(model)

def count_cnn_layers(model):
    """
    Counts the number of CNN layers in the model.

    Args:
        model (torch.nn.Module): The model whose CNN layers are to be counted.

    Returns:
        int: The number of CNN layers in the model.
    """
    return sum(1 for layer in model.modules() if isinstance(layer, nn.Conv2d))

def get_num_heads(model):
    """
    Gets the number of heads in the model.

    Args:
        model (torch.nn.Module): The model from which to get the number of heads.

    Returns:
        int: The number of heads in the model.
    """
    # For ViT, the number of heads can be directly extracted from the transformer layers
    if hasattr(model, 'transformer') and hasattr(model.transformer.layers[0][0], 'heads'):
        return model.transformer.layers[0][0].heads
    else:
        for module in model.modules():
            if hasattr(module, 'heads'):
                return module.heads
        return None

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
