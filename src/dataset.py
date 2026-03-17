from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class CheXpertDataset(Dataset):
    def __init__(self, dataframe, target_cols):
        self.df = dataframe.reset_index(drop=True).copy()

        self.target_cols = target_cols

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["Path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        target_values = torch.tensor(
            row[self.target_cols].astype(float).values,
            dtype=torch.float32,
        )

        valid_mask = (~torch.isnan(target_values)) & (target_values != -1)
        targets = torch.where(valid_mask, target_values, torch.zeros_like(target_values))

        return image, targets, valid_mask