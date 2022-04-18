import torch
from torchmetrics import IoU

target = torch.randint(0, 2, (10, 25, 25))
pred = torch.tensor(target)
pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
iou = IoU(num_classes=2)
iou(pred, target)



torchmetrics.IoU(num_classes, ignore_index=255, absent_score=0.0, threshold=0.5, 
                reduction='elementwise_mean', compute_on_step=True, dist_sync_on_step=False, 
                process_group=None)

