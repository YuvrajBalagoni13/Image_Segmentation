import torch
import torch.nn as nn

class CombineLoss(nn.Module):
  """
  Args:
      class_rgb (torch.Tensor): RGB values per class. Shape [num_classes, 3]
      alpha (float): Weight for CrossEntropy (0.5 by default)
      smooth (float): Smoothing factor for Dice (1e-6 by default)
  """
  def __init__(self, class_rgb, alpha=0.5, smooth=1e-6):
    super(CombineLoss, self).__init__()
    self.class_rgb = class_rgb.view(32,1,1,3)
    self.smooth = smooth
    self.alpha = alpha
    self.ce_loss = nn.CrossEntropyLoss()

  def _rgb_to_indices(self, target):
    matches = (target.unsqueeze(1) == self.class_rgb).all(dim=-1)
    return matches.float().argmax(dim=1)

  def _dice_loss(self, pred, target_indices):
    num_classes = pred.shape[1]
    targets_onehot = torch.eye(num_classes, device=pred.device)[target_indices]
    targets_onehot = targets_onehot.permute(0, 3, 1, 2)

    pred_probs = torch.softmax(pred, dim=1)
    intersection = torch.sum(pred_probs * targets_onehot, dim=(2, 3))
    union = torch.sum(pred_probs + targets_onehot, dim=(2, 3))

    dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
    return 1 - dice.mean()

  def forward(self, pred, target):
    """
    Args:
        pred (torch.Tensor): Model logits (unactivated). Shape [B, Num_Classes, H, W]
        target (torch.Tensor): Ground truth RGB. Shape [B, H, W, 3]
    """
    target_indices = self._rgb_to_indices(target)
    ce = self.ce_loss(pred, target_indices)
    dice = self._dice_loss(pred, target_indices)
    return self.alpha * ce + (1 - self.alpha) * dice

def rgb_to_indices(target, class_rgb):
  class_rgb = class_rgb.view(32,1,1,3)
  matches = (target.unsqueeze(1) == class_rgb).all(dim=-1)
  return matches.float().argmax(dim=1)

def Compute_IOU(pred, target, num_classes=32, smooth=1e-6):
  """
  Args:
      preds: [B, Num_Classes, H, W] (logits)
      targets: [B, H, W] (class indices)
      class_rgb: [Num_Classes, 3]
      num_classes: int
      smooth: smoothing factor
  Returns:
      iou_per_class: [Num_Classes]
      mean_iou: scalar
  """
  preds = torch.argmax(pred, dim=1)
  identity_tensor = torch.eye(num_classes).to(pred.device)
  pred_onehot = identity_tensor[preds].permute(0,3,1,2).to(pred.device)
  target_onehot = identity_tensor[target].permute(0,3,1,2).to(pred.device)

  intersection = (pred_onehot.bool() & target_onehot.bool()).float().sum(dim=(0, 2, 3))
  union = (pred_onehot.bool() | target_onehot.bool()).float().sum(dim=(0, 2, 3))

  iou_per_class = (intersection + smooth) / (union + smooth)
  mean_iou = iou_per_class[~torch.isnan(iou_per_class)].mean()

  return iou_per_class, mean_iou

def Compute_Dice(pred, target, num_classes=32, smooth=1e-6):
  """
  Args:
      preds: [B, Num_classes, H, W] (logits)
      targets: [B, H, W] (class indices)
      num_classes: int
      smooth: smoothing factor
  Returns:
      dice_per_class: [C]
      mean_dice: scalar
  """
  preds = torch.argmax(pred, dim=1)
  identity_tensor = torch.eye(num_classes).to(pred.device)
  preds_onehot = identity_tensor[preds].permute(0, 3, 1, 2).to(preds.device)
  targets_onehot = identity_tensor[target].permute(0, 3, 1, 2).to(preds.device)

  intersection = (preds_onehot * targets_onehot).float().sum(dim=(0, 2, 3))
  preds_sum = preds_onehot.float().sum(dim=(0, 2, 3))
  targets_sum = targets_onehot.float().sum(dim=(0, 2, 3))

  dice_per_class = (2.0 * intersection + smooth) / (preds_sum + targets_sum + smooth)
  mean_dice = dice_per_class[~torch.isnan(dice_per_class)].mean()

  return dice_per_class, mean_dice

def Pixel_Accuracy(pred, target):
  """
  Args:
      preds: [B, Num_classes, H, W] (logits)
      targets: [B, H, W] (class indices)
  Returns:
      Accuracy: pixelwise accuracy
  """
  preds = torch.argmax(pred, dim=1)
  total_pixel = target.numel()
  return (preds == target).sum().item() / (total_pixel) * 100