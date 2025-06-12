import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import opendatasets as od
from pathlib import Path
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def DownloadData(dataset_url: str = "https://www.kaggle.com/datasets/carlolepelaars/camvid"):
  data_path = "camvid/CamVid"

  data_dir = Path(data_path)

  if data_dir.is_dir():
    print(f"{data_dir} already exists.")
  else:
    print("Downloading data...")
    od.download(dataset_url)
  
  train_img_dir = Path(os.path.join(data_dir, "train"))
  train_mask_dir = Path(os.path.join(data_dir, "train_labels"))
  test_img_dir = Path(os.path.join(data_dir, "test"))
  test_mask_dir = Path(os.path.join(data_dir, "test_labels"))
  val_img_dir = Path(os.path.join(data_dir, "val"))
  val_mask_dir = Path(os.path.join(data_dir, "val_labels"))

  test_img_list = list(test_img_dir.glob("*.png"))
  test_mask_list = list(test_mask_dir.glob("*.png"))
  test_img_list.sort()
  test_mask_list.sort()

  for i in range(len(test_img_list) - 100):
    os.rename(test_img_list[i], os.path.join(train_img_dir, test_img_list[i].name))
    os.rename(test_mask_list[i], os.path.join(train_mask_dir, test_mask_list[i].name))
  
  return train_img_dir, train_mask_dir, val_img_dir, val_mask_dir


class CamVidDataset(Dataset):
  def __init__(self, image_dir, mask_dir, transform=None):
    self.image_dir = image_dir
    self.mask_dir = mask_dir
    self.transform = transform
    self.images = os.listdir(image_dir)
    self.masks = os.listdir(mask_dir)

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    img_path = os.path.join(self.image_dir, self.images[index])
    basename, extension = os.path.splitext(self.images[index])
    maskname = basename + "_L" + extension
    mask_path = os.path.join(self.mask_dir, maskname)
    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("RGB"))
    mask[mask == 255.0] = 1.0

    if self.transform is not None:
      augmentations = self.transform(image=image, mask=mask)
      image = augmentations["image"]
      mask = augmentations["mask"]

    return image, mask
  
def CamVidDataloader(train_img_dir,
                     train_mask_dir,
                     val_img_dir,
                     val_mask_dir,
                     batch_size):
    train_transform = A.Compose(
        [
            A.Resize(height=512, width=512),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0
            ),
            A.ToTensorV2(),
        ],
    )
    val_transform = A.Compose(
        [
            A.Resize(height=512, width=512),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0
            ),
            A.ToTensorV2(),
        ],
    )
    train_data = CamVidDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform= train_transform
    )
    val_data = CamVidDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform= val_transform
    )

    train_dataloader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True)
    val_dataloader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=False)
    return train_dataloader, val_dataloader

def CreateDataset(Dataset_URL: str = "https://www.kaggle.com/datasets/carlolepelaars/camvid",
                  batch_size: int = 2):
  
  train_img_dir, train_mask_dir, val_img_dir, val_mask_dir = DownloadData(dataset_url= Dataset_URL)
  train_dataloader, val_dataloader = CamVidDataloader(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size)
  return train_dataloader, val_dataloader