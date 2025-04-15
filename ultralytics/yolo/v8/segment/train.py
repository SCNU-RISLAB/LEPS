# Ultralytics YOLO 🚀, AGPL-3.0 license
from copy import copy

from triton.debugger.debugger import torch

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.yolo import v8
from ultralytics.yolo.utils import DEFAULT_CFG, RANK, LOGGER
from ultralytics.yolo.utils.plotting import plot_images, plot_results

import sys
sys.path.append("/home/lenovo1/project/hxn/ultralytics-main-new")
from ultralytics.yolo.utils.torch_utils import intersect_dicts

# BaseTrainer python usage
class SegmentationTrainer(v8.detect.DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'segment'
        super().__init__(cfg, overrides, _callbacks)
        self.weights=overrides['weights']
        self.freenze = overrides['freenze']

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.weights:
            ckpt = torch.load(self.weights, map_location='cpu')
            model = SegmentationModel(cfg or ckpt['model'], ch=3, nc=self.data['nc'], verbose=verbose and RANK == -1)
            csd = ckpt['model'].float().state_dict()
            csd = intersect_dicts(csd, model.state_dict())
            model.load_state_dict(csd, strict=False)
            LOGGER.info(f"Transferred {len(csd)} / {len(model.state_dict())} items from {self.weights}")
            freeze = [self.freenze]
            freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
            for k, v in model.named_parameters():
                v.requires_grad = True
                if any(x in k for x in freeze):
                    LOGGER.info(f"freezing {k}")
                    v.requires_grad = False
        else:
          model = SegmentationModel(cfg, ch=3, nc=self.data['nc'], verbose=verbose and RANK == -1)
        return model



    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss'
        return v8.segment.SegmentationValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        plot_images(batch['img'],
                    batch['batch_idx'],
                    batch['cls'].squeeze(-1),
                    batch['bboxes'],
                    batch['masks'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png

def train(cfg=DEFAULT_CFG, use_python=False):
    """Train a YOLO segmentation model based on passed arguments."""
    model = cfg.model or 'yolov8n-seg.pt'
    data = cfg.data or 'coco128-seg.yaml'  # or yolo.ClassificationDataset("mnist")
    freenze = cfg.freenze
    weights = cfg.weights
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device,weights=weights, freenze=freenze)  #weights=weights
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = SegmentationTrainer(overrides=args)
        trainer.train()


if __name__ == '__main__':
    train()
