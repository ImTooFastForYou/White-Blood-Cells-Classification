import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score


def get_model():
    # ResNet model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Adapt to my usecase
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 13)

    return model


def train_model(
    model, device, train_loader, val_loader, criterion, optimizer, num_epochs, patience
):
    best_f1 = 0
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%"
        )

        # Early Stopping
        current_f1 = validate_model(model, device, val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Macro-F1: {current_f1:.4f}")

        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), "best_white_cell_model.pth")
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early Stopping !")
            break

    print("Training finished !")


def validate_model(model, device, test_loader):
    model.eval()
    all_pred = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_pred.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    score = f1_score(all_labels, all_pred, average="macro")
    return score


def predict_model(model, device, test_loader, inv_label_map):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

    return [inv_label_map[p] for p in all_preds]
