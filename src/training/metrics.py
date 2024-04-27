import torch
from . import base
from . import functional as F
from . import _modules as modules

from torchmetrics import JaccardIndex

class Jacard(base.Metric):
    __name__ = "jacard_index"

    def __init__(self, task, threshold=None, activation=None, ignore_index=None,
                 num_classes=None, num_labels=None, average='macro', validate_args=True, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = modules.Activation(activation, dim=1)
        self.task = task
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.average = average
        self.validate_args = validate_args

        self.jaccard = JaccardIndex(task=self.task, 
                                    num_classes=self.num_classes, 
                                    threshold=self.threshold,
                                    num_labels=self.num_labels,
                                    average=self.average,
                                    ignore_index=self.ignore_index,
                                    validate_args=self.validate_args)

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return self.jaccard(y_pr, y_gt)


class IoU(base.Metric):
    __name__ = "iou"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, drop_empty=False, take_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = modules.Activation(activation, dim=1)
        self.ignore_channels = ignore_channels
        self.per_image = per_image
        self.class_weights = class_weights
        self.drop_empty = drop_empty
        self.take_channels = take_channels

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
            drop_empty=self.drop_empty,
            take_channels=self.take_channels,
        )


class MicroIoU(base.Metric):
    __name__ = "micro_iou"

    def __init__(self, threshold=0.5):
        super().__init__()
        self.eps = 1e-5
        self.intersection = 0.
        self.union = 0.
        self.threshold = threshold

    def reset(self):
        self.intersection = 0.
        self.union = 0.

    @torch.no_grad()
    def __call__(self, prediction, target):
        prediction = (prediction > self.threshold).float()

        intersection = (prediction * target).sum()
        union = (prediction + target).sum() - intersection

        self.intersection += intersection.detach()
        self.union += union.detach()

        score = (self.intersection + self.eps) / (self.union + self.eps)
        return score

class DiceCoef(base.Metric):
    __name__ = 'dice_coef'

    def __init__(self, eps=1e-7, beta=1., threshold=0.5 ,ignore_channels=None,
                 per_image=False, class_weights=None, drop_empty=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.ignore_channels = ignore_channels
        self.threshold = threshold
        self.per_image = per_image
        self.class_weights = class_weights
        self.drop_empty = drop_empty

    @torch.no_grad()
    def __call__(self, prediction, target):
        return F.f_score(
            prediction, target,
            beta=self.beta,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
            drop_empty=self.drop_empty,
        )
    
class DummyMetric(base.Metric):
    __name__ = 'dummy_metric'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return torch.tensor(1)