# Compute and plot the confusion matrix
y_true = []  # True labels
y_pred = []  # Predicted labels
# Set the model to evaluation mode
model.eval()
for data in test_loader:
    data = data.to(device)
    with torch.no_grad():
        out_test = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred_test = out_test.argmax(dim=1).cpu().numpy()
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(pred_test)
# Compute confusion matrix
confusion = confusion_matrix(y_true, y_pred)
# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(confusion, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
# Replace 'Class 0', 'Class 1', 'Class 2', etc. with your actual class labels
classes = ['0', '1','2','3','4','5','6','7','8','9']  # Replace with your class labels
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
# Display values inside the matrix
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(confusion[i][j]), ha='center', va='center', color='white' if confusion[i][j] > confusion.max() / 2 else 'black')
 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()