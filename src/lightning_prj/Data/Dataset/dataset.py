from pathlib import Path

import lightning as ptl
import polars as pl
import torch
import torchvision.io as tv_io
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from ...Config.config import HYPER_PARAMS


class FoodDataset(Dataset):
    def __init__(self, split) -> None:
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of 'train', 'val', or 'test'.")
        self.root_path = Path(f"../Dataset/Split/{split}").resolve()
        self.len = len(list(self.root_path.glob("*"))) - 1
        full_df = pl.read_csv(self.root_path.parent / "split_info.csv")
        unique_class = full_df.select(pl.col("class")).unique().sort(pl.all())
        self.class_mapping = {
            class_name: idx for idx, class_name in enumerate(unique_class["class"])
        }
        self.df = full_df.filter(pl.col("split") == split)
        self.transform = v2.Compose([
            v2.RandomResizedCrop((224), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(degrees=(-15, 15)),
            v2.RandomAffine(
                degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1
            ),
            v2.RandomInvert(p=0.5),
            v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
            v2.RandomAutocontrast(),
            v2.RandomEqualize(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        file_info = self.df[index]
        file_name = self.root_path / f"{file_info['class'].item()}_{file_info['file_name'].item()}"
        file_tensor = tv_io.decode_image(str(file_name))
        file_class = self.class_mapping[file_info["class"].item()]
        transform_image = self.transform(file_tensor)
        return transform_image, file_class


class FoodDataModule(ptl.LightningDataModule):
    def __init__(self, config=HYPER_PARAMS) -> None:
        super().__init__()
        self.config = config
        self.batch_size = config.BATCH_SIZE

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = FoodDataset("train")
            self.val_dataset = FoodDataset("val")
        if stage == "test" or stage is None:
            self.test_dataset = FoodDataset("test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )
