import torch.nn as nn
import torch.optim as optim
import torch
import os
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from sklearn.model_selection import KFold
from models.ViT_early_exit import ViT_early_exit
from models.ViT_CNN_early_exit import ViT_CNN_early_exit
from functions.data_loader import (
    load_checkpoint,
    save_best_model,
    save_lists_to_file,
)
from functions.helpers import (
    compute_pruning_ammount,
    feature_distillation_loss,
    knowledge_distillation_loss,
)
from functions.plotter import plot_loss_accuracy


def test(model, test_loader, device, feature_kd=None):
    model.to(device)
    model.eval()
    list_early_exit_info = []
    correct = 0
    total = 0
    num_early_exits=0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if feature_kd == 'feature' or feature_kd == 'combined':
                outputs, _,__ = model(inputs)
            elif isinstance(model,ViT_early_exit) or isinstance(model,ViT_CNN_early_exit):
                outputs, early_exit_info = model(inputs)
                list_early_exit_info.append(early_exit_info)
                if early_exit_info['exited']:  # Check if early exit occurred
                    num_early_exits += 1
            else:
                outputs, _= model(inputs)
            # _, predicted = torch.max(outputs.sup, 1)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    if isinstance(model,ViT_early_exit) or isinstance(model,ViT_CNN_early_exit):
        print(f"Number of early exits: {num_early_exits}")
        return accuracy,num_early_exits #,list_early_exit_info
    return accuracy


def test_batch(model, image_batch, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = image_batch.to(device)
        outputs, _ = model(inputs)
        _, predicted = torch.max(outputs, 1)

    return predicted


def train(
    model,
    train_loader,
    test_loader,
    epochs,
    learning_rate,
    device,
    pruning_method=None,
    weight_decay=0.0005,
    save_path=None,
    load_path=None,
):
    if load_path:
        checkpoint = load_checkpoint(model, load_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # make weights sparse tensors
    # if pruning_method:
    #     for param in model.parameters():
    #         param = nn.Parameter(torch.sparse_coo_tensor(param.shape).to("cuda"))

    model.train()
    max_test_accuracy = 0
    loss_list = []
    accuracy_list = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                outputs, _ = model(
                    inputs
                )  # cambiado el outputs a outputs[0] para probar kd
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        lr_scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
        print("Current Learning Rate:", optimizer.param_groups[0]["lr"])

        _, predicted = outputs.max(1)  # cambiado el outputs a outputs[0] para probar kd
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total
        print(f"accuracy: {accuracy}%")
        if isinstance(model,ViT_early_exit) or isinstance(model,ViT_CNN_early_exit):
            test_accuracy,_ = test(model, test_loader, device)
        else:
            test_accuracy = test(model, test_loader, device)
        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            print(f"max_test_accuracy : {max_test_accuracy}")
            if save_path:
                save_best_model(
                    model, optimizer, lr_scheduler, max_test_accuracy, save_path
                )
        loss_list.append(epoch_loss)
        accuracy_list.append(test_accuracy)
        if pruning_method:
            pruning_amount = compute_pruning_ammount(epoch)
            if pruning_amount > 0:
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        if pruning_method == "structured":
                            prune.ln_structured(
                                module, name="weight", amount=pruning_amount, n=2, dim=0
                            )
                        elif pruning_method == "unstructured":
                            prune.ln_unstructured(
                                module, name="weight", amount=pruning_amount, n=2, dim=0
                            )

    save_lists_path = f"{save_path}/loss_and_accuracy"
    print(f"max_training_accuracy : {max_test_accuracy}")
    plot_loss_accuracy(loss_list, accuracy_list, "model")
    save_lists_to_file(loss_list, accuracy_list, save_path=save_lists_path)


def train_kd(
    student,
    teacher,
    train_loader,
    test_loader,
    epochs,
    learning_rate,
    device,
    T,
    alpha,
    beta=None,
    distill_mode=None,
    weight_decay=0.0005,
    save_path=None,
    load_path_teacher=None,
):
    if load_path_teacher:
        load_checkpoint(teacher, load_path_teacher)
    criterion = nn.CrossEntropyLoss()
    # optimizer = build_optimizer(model,learning_rate)
    optimizer = optim.Adam(
        student.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # make weights sparse tensors
    # if pruning_method:
    #     for param in student.parameters():
    #         param = nn.Parameter(torch.sparse_coo_tensor(param.shape).to("cuda"))
    max_test_accuracy = 0
    teacher.eval()
    student.train()
    loss_list = []
    accuracy_list = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                if distill_mode == 'feature':
                    teacher_logits, _, teacher_features = teacher(inputs)
                elif distill_mode == 'response':
                    teacher_logits, _ = teacher(inputs)
                elif distill_mode == 'combined':
                    teacher_logits, _, teacher_features = teacher(inputs)

            with torch.cuda.amp.autocast(enabled=True):
                if distill_mode == 'feature':
                    student_logits, _, student_features = student(inputs)
                    distill_loss = feature_distillation_loss(
                        student_features, teacher_features
                    )
                    # distill_loss_2 = knowledge_distillation_loss(
                    #     student_logits, teacher_logits, T
                    # )
                elif distill_mode == 'response':
                    student_logits, _ = student(inputs)
                    distill_loss = knowledge_distillation_loss(
                        student_logits, teacher_logits, T
                    )
                elif distill_mode == 'combined':
                    student_logits, _, student_features = student(inputs)
                    feature_distill_loss = feature_distillation_loss(
                        student_features, teacher_features
                    )
                    response_distill_loss = knowledge_distillation_loss(
                        student_logits, teacher_logits, T
                    )

                label_loss = criterion(student_logits, labels)
                if distill_mode == 'combined':
                    loss = alpha * feature_distill_loss + beta * response_distill_loss+ (1 - (alpha-beta)) * label_loss 
                else:
                    loss = alpha * distill_loss + (1 - alpha) * label_loss 

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        lr_scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
        print("Current Learning Rate:", optimizer.param_groups[0]["lr"])

        # if pruning_method:
        #     pruning_amount = compute_pruning_ammount(epoch)
        #     if pruning_amount > 0:
        #         for name, module in student.named_modules():
        #             if isinstance(module, nn.Linear):
        #                 if pruning_method == "structured":
        #                     prune.ln_structured(
        #                         module, name="weight", amount=pruning_amount, n=2, dim=0
        #                     )
        #                 elif pruning_method == "unstructured":
        #                     prune.ln_unstructured(
        #                         module, name="weight", amount=pruning_amount, n=1, dim=0
        #                     )

        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total
        print(f"accuracy: {accuracy}%")
        test_accuracy = test(student, test_loader, device, distill_mode)
        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            print(f"max_test_accuracy : {max_test_accuracy}")
            if save_path:
                save_best_model(
                    student, optimizer, lr_scheduler, test_accuracy, save_path
                )

        loss_list.append(epoch_loss)
        accuracy_list.append(test_accuracy)

    print(f"max_training_accuracy : {max_test_accuracy}")
    plot_loss_accuracy(loss_list, accuracy_list, "model")
    if save_path:
        save_lists_path = f"{save_path}/loss_and_accuracy"
        save_lists_to_file(loss_list, accuracy_list, save_path=save_lists_path)


def train_feature_kd(
    student,
    teacher,
    train_loader,
    test_loader,
    T,
    alpha,
    epochs,
    learning_rate,
    device,
    weight_decay,
    hard=False,
    save_path=None,
    load_path_teacher=None,
):
    if load_path_teacher:
        load_checkpoint(teacher, load_path_teacher)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        student.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    max_test_accuracy = 0
    teacher.eval()
    student.train()
    loss_list = []
    accuracy_list = []

    min_layers = min(len(teacher.transformer.layers), len(student.transformer.layers))
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits, _ = teacher(inputs)

            student_logits, distill_logits = student(inputs)

            soft_targets = nn.functional.log_softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
            soft_targets_loss = (
                torch.sum(soft_targets * (soft_targets - soft_prob))
                / soft_prob.size(0)
                * (T**2)
            )

            # feature_loss = 0.0
            # for teacher_feat, student_feat in zip(
            #     teacher_intermediate[:min_layers], student_intermediate[:min_layers]
            # ):
            #     feature_loss += nn.functional.mse_loss(student_feat, teacher_feat)

            label_loss = criterion(student_logits, labels)
            if not hard:
                distill_loss = F.kl_div(
                    F.log_softmax(distill_logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean",
                )
                distill_loss *= T**2
            else:
                teacher_labels = teacher_logits.argmax(dim=-1)
                distill_loss = F.cross_entropy(distill_logits, teacher_labels)

            loss = (
                (1 - alpha) * soft_targets_loss
                + distill_loss * alpha
                # + feature_loss
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        lr_scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
        print("Current Learning Rate:", optimizer.param_groups[0]["lr"])

        # Compute accuracy
        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total
        print(f"accuracy: {accuracy}%")
        test_accuracy = test(student, test_loader, device)

        # Update max test accuracy and save model if necessary
        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            print(f"max_test_accuracy : {max_test_accuracy}")
            if save_path:
                save_best_model(
                    student, optimizer, lr_scheduler, test_accuracy, save_path
                )

        loss_list.append(epoch_loss)
        accuracy_list.append(test_accuracy)

    print(f"max_training_accuracy : {max_test_accuracy}")
    plot_loss_accuracy(loss_list, accuracy_list, "model")
    if save_path:
        save_lists_path = f"{save_path}/loss_and_accuracy"
        save_lists_to_file(loss_list, accuracy_list, save_path=save_lists_path)


def test_e(model, test_loader, device, feature_kd=False):
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if not feature_kd:
                outputs = model(inputs)
            # _, predicted = torch.max(outputs.sup, 1)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def train_e(
    model,
    train_loader,
    test_loader,
    epochs,
    learning_rate,
    device,
    pruning_method=None,
    weight_decay=0.0005,
    save_path=None,
    load_path=None,
):
    if load_path:
        checkpoint = load_checkpoint(model, load_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # make weights sparse tensors
    # if pruning_method:
    #     for param in model.parameters():
    #         param = nn.Parameter(torch.sparse_coo_tensor(param.shape).to("cuda"))

    model.train()
    max_test_accuracy = 0
    loss_list = []
    accuracy_list = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                outputs= model(
                    inputs
                )  # cambiado el outputs a outputs[0] para probar kd
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        lr_scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
        print("Current Learning Rate:", optimizer.param_groups[0]["lr"])

        _, predicted = outputs.max(1)  # cambiado el outputs a outputs[0] para probar kd
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total
        print(f"accuracy: {accuracy}%")

        test_accuracy = test(model, test_loader, device)
        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            print(f"max_test_accuracy : {max_test_accuracy}")
            if save_path:
                save_best_model(
                    model, optimizer, lr_scheduler, max_test_accuracy, save_path
                )
        loss_list.append(epoch_loss)
        accuracy_list.append(test_accuracy)
        if pruning_method:
            pruning_amount = compute_pruning_ammount(epoch)
            if pruning_amount > 0:
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        if pruning_method == "structured":
                            prune.ln_structured(
                                module, name="weight", amount=pruning_amount, n=2, dim=0
                            )
                        elif pruning_method == "unstructured":
                            prune.ln_unstructured(
                                module, name="weight", amount=pruning_amount, n=2, dim=0
                            )

    save_lists_path = f"{save_path}/loss_and_accuracy"
    print(f"max_training_accuracy : {max_test_accuracy}")
    plot_loss_accuracy(loss_list, accuracy_list, "model")
    save_lists_to_file(loss_list, accuracy_list, save_path=save_lists_path)
