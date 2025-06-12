import torch
from tqdm.auto import tqdm
import onnx
from torch.amp import autocast, GradScaler
import wandb
import pandas as pd
from pathlib import Path

from Dataset.CamVidDataset import CreateDataset
from Models import R50AttentionUNet, R50UNet, AttentionUNet, UNet
from Metrics.metrics import CombineLoss, Compute_Dice, Compute_IOU, rgb_to_indices, Pixel_Accuracy

def Train(architecture: str = "R50AttentionUNet",
          lr: float = 0.001,
          epochs: int = 100,
          batch_size: int = 2,
          loss_alpha: float = 0.5,
          early_stopping_patience: int = 3):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "Architecture": architecture,
        "LR": lr,
        "Epochs": epochs,
        "Batch_Size": batch_size,
        "loss_alpha": loss_alpha,
        "Early_Stopping_Patience": early_stopping_patience,
    }

    run = wandb.init(project="Image_Segmentation", config=config)
    config = run.config

    train_dataloader, val_dataloader = CreateDataset(batch_size= config.Batch_Size)

    if config.Architecture == "R50AttentionUNet":
        Model = R50AttentionUNet.R50_AttentionUNet(in_channels = 3,
                                                   out_channels = 32).to(device)
    elif config.Architecture == "R50UNet":
        Model = R50UNet.R50UNet(in_channels = 3,
                                out_channels = 32).to(device)
    elif config.Architecture == "AttentionUNet":
        Model = AttentionUNet.AttentionUNet(in_channels = 3,
                                            out_channels = 32).to(device)
    elif config.Architecture == "UNet":
        Model = UNet.UNet(in_channels = 3,
                          out_channels = 32).to(device)


    result = {
        "train loss": [],
        "train acc" : [],
        "train miou": [],
        "train dice": [],
        "val loss": [],
        "val acc" : [],
        "val miou": [],
        "val dice": []
    }
    class_dict = pd.read_csv("camvid/CamVid/class_dict.csv")

    class_rgb = torch.tensor(class_dict.iloc[:, 1:].values, dtype=torch.float64).to(device)

    loss_fn = CombineLoss(class_rgb = class_rgb, alpha = config.loss_alpha)
    optimizer = torch.optim.Adam(Model.parameters(),
                                lr = config.LR)

    prev_loss = float("inf")
    patience = 0

    for epoch in tqdm(range(config.Epochs)):
        Model.train()
        train_loss, train_acc, train_miou, train_dice = 0, 0, 0, 0

        scaler = GradScaler(device=device)
        for batch_idx, (x,y) in enumerate(train_dataloader):
            x,y = x.to(device), y.to(device)
            with autocast(device):
                y_pred = Model(x).to(device)
                loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            y = rgb_to_indices(y, class_rgb)

            _, mean_iou = Compute_IOU(y_pred, y)
            _, mean_dice = Compute_Dice(y_pred, y)
            accuracy = Pixel_Accuracy(y_pred, y)

            train_miou += mean_iou
            train_dice += mean_dice
            train_acc += accuracy

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        train_miou /= len(train_dataloader)
        train_dice /= len(train_dataloader)

        Model.eval()
        val_loss, val_acc, val_miou, val_dice = 0, 0, 0, 0

        with torch.inference_mode():
            for batch_idx, (x,y) in enumerate(val_dataloader):
                x,y = x.to(device), y.to(device)

                with autocast(device):
                    y_pred = Model(x).to(device)
                    loss = loss_fn(y_pred, y)
                val_loss += loss.item()

                y = rgb_to_indices(y, class_rgb)

                _, mean_iou = Compute_IOU(y_pred, y)
                _, mean_dice = Compute_Dice(y_pred, y)
                accuracy = Pixel_Accuracy(y_pred, y)

                val_miou += mean_iou
                val_dice += mean_dice
                val_acc += accuracy

            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)
            val_miou /= len(val_dataloader)
            val_dice /= len(val_dataloader)

        result["train loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        result["train acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        result["train miou"].append(train_miou.item() if isinstance(train_miou, torch.Tensor) else train_miou)
        result["train dice"].append(train_dice.item() if isinstance(train_dice, torch.Tensor) else train_dice)
        result["val loss"].append(val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss)
        result["val acc"].append(val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc)
        result["val miou"].append(val_miou.item() if isinstance(val_miou, torch.Tensor) else val_miou)
        result["val dice"].append(val_dice.item() if isinstance(val_dice, torch.Tensor) else val_dice)

        run.log(
            {
                "epoch": epoch + 1,
                "train loss": train_loss,
                "train miou": train_miou,
                "train dice": train_dice,
                "val loss": val_loss,
                "val miou": val_miou,
                "val dice": val_dice
            }
        )

        print(f"Epoch {epoch + 1}/{config.Epochs}| train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | train miou: {train_miou:.4f} | train dice: {train_dice:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.4f} | val miou: {val_miou:.4f} | val dice: {val_dice:.4f}")

        min_depth = 0.002
        if val_loss > prev_loss - min_depth:
            patience += 1
            if patience >= config.Early_Stopping_Patience:
                print(f"No improvement in validation loss so stopping training at {epoch}/{config.Epochs}\n")
                break
        else:
            patience = 0
        prev_loss = min(prev_loss, val_loss)

    model_path = Path("Trained_Models/")
    if model_path.is_dir() == False:
        model_path.mkdir()

    torch.onnx.export(
        Model,
        torch.randn(1,3,512,512, device=device),
        "Trained_Models/model.onnx",
        input_names=["input"],
        output_names=["output"],
    )

    run.log_artifact("Trained_Models/model.onnx", type = "Model")
    print("Model training Completed")
    run.finish()

    return result
