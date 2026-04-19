from data_pre_treatment import *
from white_cells_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
model.load_state_dict(
    torch.load("best_white_cell_model.pth", map_location=torch.device("cpu"))
)
train_csv = "Data/train_metadata.csv"
test_csv = "Data/test_metadata.csv"
train_dir = "Data/train/"
test_dir = "Data/test/"
train_loader, val_loader, test_loader = load_data(
    train_csv, test_csv, train_dir, test_dir
)
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
sample_sub = pd.read_csv("Data/sample_submission.csv")
if len(sample_sub) == len(predictions):
    sample_sub["label"] = predictions
    sample_sub.to_csv("submission.csv", index=False)
    print("Fichier submission.csv prêt !")
else:
    print(f"Size error ! Got: {len(predictions)} instead of: {len(sample_sub)}")
print("Model tested !")
