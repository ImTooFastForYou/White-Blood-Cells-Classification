import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split


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
    if "label" in df.columns:
        df["label_idx"] = df["label"].map(label_map)
    else:
        df["label"] = "undefined"
        df["label_idx"] = -1
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


def load_data(train_df_path, test_df_path, train_data_path, test_data_path):
    # Resize to 224x224 for ResNet + Data Augmentation
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    df_full_train = _to_dataframe(train_df_path)
    df_test = _to_dataframe(test_df_path)

    df_train, df_val = train_test_split(
        df_full_train,
        test_size=0.20,
        random_state=0,
        stratify=df_full_train["label_idx"],
    )

    # Datasets
    train_set = WhiteBloodCellDataset(
        dataframe=df_train,
        path=train_data_path,
        transform=train_transforms,
    )
    val_set = WhiteBloodCellDataset(
        dataframe=df_val,
        path=train_data_path,
        transform=val_test_transforms,
    )
    test_set = WhiteBloodCellDataset(
        dataframe=df_test,
        path=test_data_path,
        transform=val_test_transforms,
    )
    # Sampler
    counts = df_train["label_idx"].value_counts().sort_index().values
    weights = 1.0 / counts
    samples_weights = torch.from_numpy(weights[df_train["label_idx"].values]).double()

    sampler = WeightedRandomSampler(
        weights=samples_weights, num_samples=len(samples_weights), replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=64, sampler=sampler, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
