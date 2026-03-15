import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _to_dataframe(path):
    df = pd.read_csv(path)
    label_map = {
        "SNE": 0,
        "LY": 1,
        "MO": 2,
        "EO": 3,
        "BA": 4,
        "VLY": 5,
        "BNE": 6,
        "MMY": 7,
        "MY": 8,
        "PMY": 9,
        "BL": 10,
        "PC": 11,
        "PLY": 12,
    }
    if path.split("/")[1] == "test":
        df["label"] = "undefined"

    df["label_idx"] = df["label"].map(
        lambda x: -1 if x == "undefined" else label_map[x]
    )
    return df


class WhiteBloodCellDataset(Dataset):
    def __init__(self, dataframe, path, transform=None):
        self.names = dataframe["ID"].values
        self.labels = dataframe["label_idx"].values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        image_path = self.path + str(self.names[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def load_data(train_path, test_path):
    # Resize to 224x224 for ResNet
    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Datasets
    train_set = WhiteBloodCellDataset(
        dataframe=_to_dataframe(train_path),
        path="/Data/train/",
        transform=data_transforms,
    )
    test_set = WhiteBloodCellDataset(
        dataframe=_to_dataframe(test_path),
        path="/Data/test/",
        transform=data_transforms,
    )

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    return train_loader, test_loader
