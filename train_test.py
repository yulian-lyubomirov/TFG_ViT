import torch.nn as nn
import torch.optim as optim
import torch
import os


from helpers import (
    load_checkpoint,
    save_best_model,
    plot_loss_accuracy,
    save_lists_to_file,
)


def train(
    model,
    train_loader,
    test_loader,
    epochs,
    learning_rate,
    device,
    save_path=None,
    load_path=None,
):
    if load_path:
        checkpoint = load_checkpoint(model, load_path)
    criterion = nn.CrossEntropyLoss()
    # optimizer = build_optimizer(model,learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=200)
    model.train()
    max_training_accuracy = 0
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
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()

        lr_scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
        print("Current Learning Rate:", optimizer.param_groups[0]["lr"])
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total
        print(f"accuracy: {accuracy}%")

        if accuracy > max_training_accuracy:
            max_training_accuracy = accuracy
            print(f"max_accuracy : {max_training_accuracy}")
            if save_path:
                save_best_model(model, optimizer, lr_scheduler, accuracy, save_path)
        loss_list.append(epoch_loss)
        accuracy_list.append(accuracy)

        # if epoch % 10 == 0:
        #     save_checkpoint(epoch, model, optimizer, lr_scheduler, accuracy, save_path)
    save_lists_path = f"{save_path}/loss_and_accuracy"
    print(f"max_training_accuracy : {max_training_accuracy}")
    plot_loss_accuracy(loss_list, accuracy_list)
    save_lists_to_file(loss_list, accuracy_list, save_path=save_lists_path)


def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            # _, predicted = torch.max(outputs.sup, 1)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def train_kd(
    student,
    teacher,
    train_loader,
    test_loader,
    T,
    soft_target_loss_weight,
    ce_loss_weight,
    epochs,
    learning_rate,
    device,
    save_path=None,
    load_path_teacher=None,
):
    if load_path_teacher:
        load_checkpoint(teacher, load_path_teacher)
    criterion = nn.CrossEntropyLoss()
    # optimizer = build_optimizer(model,learning_rate)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=200)
    max_training_accuracy = 0
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
                teacher_logits = teacher(inputs)

            with torch.cuda.amp.autocast(enabled=True):
                student_logits = student(inputs)
                soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

                soft_targets_loss = (
                    torch.sum(soft_targets * (soft_targets.log() - soft_prob))
                    / soft_prob.size()[0]
                    * (T**2)
                )

                label_loss = criterion(student_logits, labels)

                loss = (
                    soft_target_loss_weight * soft_targets_loss
                    + ce_loss_weight * label_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()

        lr_scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
        print("Current Learning Rate:", optimizer.param_groups[0]["lr"])

        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total
        print(f"accuracy: {accuracy}%")

        if accuracy > max_training_accuracy:
            max_training_accuracy = accuracy
            print(f"max_accuracy : {max_training_accuracy}")
            if save_path:
                save_best_model(student, optimizer, lr_scheduler, accuracy, save_path)
        loss_list.append(epoch_loss)
        accuracy_list.append(accuracy)
        # if epoch % 10 == 0:
        #     save_checkpoint(
        #         epoch, student, optimizer, lr_scheduler, accuracy, save_path
        #     )
    print(f"max_training_accuracy : {max_training_accuracy}")
    plot_loss_accuracy(loss_list, accuracy_list)
    save_lists_path = f"{save_path}/loss_and_accuracy"
    save_lists_to_file(loss_list, accuracy_list, save_path=save_lists_path)
