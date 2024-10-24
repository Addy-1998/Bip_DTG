
#Train and test loop for Graph Attention Networks (GAT)
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
epoch = 200
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.NLLLoss()
reduce_lr_epochs = 50  # Define the number of epochs after which to reduce LR
gamma = 0.5  # Define the LR reduction factor
train_acc_history = []
test_acc_history = []
train_mAP_history = []
test_mAP_history = []
lr_history = []  # List to store LR changes
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by gamma every reduce_lr_epochs epochs."""
    lr = 0.001 * (gamma ** (epoch // reduce_lr_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def train_and_test(epoch):
    model.train()
    correct = 0
    total = 0
    y_true_train = []
    y_scores_train = []
    for data in train_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Calculate training accuracy
        with torch.no_grad():
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.y.size(0)
            pred_probs = F.softmax(out, dim=1)
            y_scores_train.extend(pred_probs.cpu().tolist())
            y_true_train.extend(data.y.cpu().tolist())
    train_accuracy = correct / total
    mAPs_train = []
    num_classes = len(y_scores_train[0])
    for class_idx in range(num_classes):
        y_true_class = [int(label == class_idx) for label in y_true_train]
        y_scores_class = [scores[class_idx] for scores in y_scores_train]
        mAP_class = average_precision_score(y_true_class, y_scores_class)
        mAPs_train.append(mAP_class)
    mean_mAP_train = sum(mAPs_train) / len(mAPs_train)
    if epoch % 5 == 0:  # Perform testing every 5 epochs
        train_acc_history.append(train_accuracy)
        train_mAP_history.append(mean_mAP_train)
        # Testing
        model.eval()
        correct_test = 0
        total_test = 0
        y_true_test = []
        y_scores_test = []
        for data in test_loader:
            data = data.to(device)
            with torch.no_grad():
                out_test = model(data.x, data.edge_index, data.edge_attr, data.batch)
                pred_test = out_test.argmax(dim=1)
                correct_test += int((pred_test == data.y).sum())
                total_test += data.y.size(0)
                pred_probs_test = F.softmax(out_test, dim=1)
                y_scores_test.extend(pred_probs_test.cpu().tolist())
                y_true_test.extend(data.y.cpu().tolist())
        test_accuracy = correct_test / total_test
        mAPs_test = []
        num_classes_test = len(y_scores_test[0])
        for class_idx in range(num_classes_test):
            y_true_class_test = [int(label == class_idx) for label in y_true_test]
            y_scores_class_test = [scores[class_idx] for scores in y_scores_test]
            mAP_class_test = average_precision_score(y_true_class_test, y_scores_class_test)
            mAPs_test.append(mAP_class_test)
        mean_mAP_test = sum(mAPs_test) / len(mAPs_test)
        test_acc_history.append(test_accuracy)
        test_mAP_history.append(mean_mAP_test)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch:03d}, Train Accuracy: {train_accuracy:.4f}, LR: {current_lr:.6f}, Test Accuracy: {test_accuracy:.4f}, Train mAP: {mean_mAP_train:.4f}, Test mAP: {mean_mAP_test:.4f}')
    # Save the LR change
    if epoch % reduce_lr_epochs == 0:
        lr_history.append(current_lr)
        print(f'Learning rate changed to: {current_lr:.6f}')
 
# Train and test loop
for epoch in range(1, epoch):
    adjust_learning_rate(optimizer, epoch)  # Adjust learning rate
    train_and_test(epoch)
# Print LR history
print("Learning Rate History:")
for epoch, lr in enumerate(lr_history):
    print(f'Epoch {epoch * reduce_lr_epochs}: {lr:.6f}')
 
from plots import plot_confusion_matrix
max(test_acc_history)