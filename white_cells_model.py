import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


def get_model():
    # ResNet model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Adapt to my usecase
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 13)

    return model


def train_model(model, device, train_loader, criterion, optimizer, num_epochs):
    model.train()

    for epoch in range(num_epochs):
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
            f"Époque [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%"
        )

    print("Entraînement terminé !")


def test_model(model, device, test_loader):
    model.eval()
    all_pred = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_pred.extend(predicted.cpu().numpy())
    return all_pred
