from data_pre_treatment import *
from white_cells_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using : {device}")

# Loading data
train_csv = "Data/train_metadata.csv"
test_csv = "Data/test_metadata.csv"
train_dir = "Data/train/"
test_dir = "Data/test/"
train_loader, val_loader, test_loader = load_data(
    train_csv, test_csv, train_dir, test_dir
)
print("Data loaded !")

# Model
model = get_model()
model = model.to(device)

# Parameters
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30
patience = 5

# Train
train_model(
    model, device, train_loader, val_loader, criterion, optimizer, num_epochs, patience
)
model.load_state_dict(torch.load("best_white_cell_model.pth"))
print("Model saved !")

# Validation
score = validate_model(model, device, val_loader)
print(f"The Macro F1-score for Validation is {score}.")


# Predictions
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
inv_label_map = {v: k for k, v in label_map.items()}
predictions = predict_model(model, device, test_loader, inv_label_map)
test_df = pd.read_csv(test_csv)
test_df["label"] = predictions
test_df.to_csv("test_prediction.csv", index=False)
print("Model tested !")
