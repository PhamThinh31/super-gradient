import sys
sys.path.append("/home/ubuntu/Workspace/thinhplg-dev/object_detection/super-gradients/src")
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
import torch
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val)


from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095, DetectionMetrics_050_5Classes, DetectionMetrics_050_3Main_Class
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset
from super_gradients.training.transforms.transforms import DetectionMosaic, DetectionRandomAffine, DetectionHSV, \
    DetectionHorizontalFlip, DetectionPaddedRescale, DetectionStandardize, DetectionTargetsFormatTransform
from super_gradients.training.utils.detection_utils import CrowdDetectionCollateFN
from super_gradients.training import dataloaders
from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed

from clearml import Task



EXPERIMENT_NAME  = 'v1.0_yolo_nas_m_od_bdd'
DATASET_VERSION = 'v1.2.0'

task = Task.init(project_name='SVM-2d-Object-Detection', task_name = EXPERIMENT_NAME)

params ={
    "optimizer": "SGD",
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer_params": {"weight_decay": 0.0001, "momentum": 0.9, "nesterov": True},
    # "optimizer_params": {"weight_decay": 0.0001},
    "ema_params": {"decay": 0.9997, "decay_type": "threshold"},
    "max_epochs": 50,
    "resume": False,
}
task.connect(params)

from super_gradients.common.data_types.enum import MultiGPUMode

setup_device(multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, num_gpus=4)


# DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_ARCH = 'yolo_nas_m'

# model = models.get(MODEL_ARCH, pretrained_weights="coco").to(DEVICE)

HOME = '/home/ubuntu/Workspace/thinhplg-dev/object_detection/super-gradients'
BATCH_SIZE = 16

CHECKPOINT_DIR = f'{HOME}/checkpoints'



trainer = Trainer(experiment_name=EXPERIMENT_NAME, ckpt_root_dir=CHECKPOINT_DIR)


CLASSES = ['car', 'pedestrians', 'bicycle', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'cat', 'dog', 'backpack', 'umbrella', 'handbag', 'suitcase', 'snowboard', 'sports ball', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'clock', 'vase']


trainset = COCOFormatDetectionDataset(data_dir="/home/ubuntu/Workspace/thinhplg-dev/datasets/bdd100k/tracking/bdd100k",
                                      images_dir="",
                                      json_annotation_file=os.path.join("tracking/v1.0_train_full.json"),
                                      
                                      input_dim=(640, 640),
                                      ignore_empty_annotations=False,
                                      transforms=[
                                          DetectionMosaic(prob=1., input_dim=(640, 640)),
                                          DetectionRandomAffine(degrees=0., scales=(0.5, 1.5), shear=0.,
                                                                target_size=(640, 640),
                                                                filter_box_candidates=False, border_value=128),
                                          DetectionHSV(prob=1., hgain=5, vgain=30, sgain=30),
                                          DetectionHorizontalFlip(prob=0.5),
                                          DetectionPaddedRescale(input_dim=(640, 640), max_targets=300),
                                          DetectionStandardize(max_value=255),
                                        #   DetectionMixup('additional_samples_count': 1, 'non_empty_targets': True, 'input_dim': [640, 640], 'mixup_scale': [0.5, 1.5], 'prob': 1.0, 'enable_mixup': True, 'flip_prob': 0.5, 'border_value': 114),
                                          DetectionTargetsFormatTransform(max_targets=300, input_dim=(640, 640),
                                                                          output_format="LABEL_CXCYWH")
                                      ])



valset = COCOFormatDetectionDataset(data_dir="/home/ubuntu/Workspace/thinhplg-dev/datasets/bdd100k/tracking/bdd100k",
                                    images_dir="",
                                    json_annotation_file=os.path.join("tracking/v1.0_val_full.json"),
                                    
                                    input_dim=(640, 640),
                                    ignore_empty_annotations=False,
                                    transforms=[
                                        DetectionPaddedRescale(input_dim=(640, 640), max_targets=300),
                                        DetectionStandardize(max_value=255),
                                        DetectionTargetsFormatTransform(max_targets=300, input_dim=(640, 640),
                                                                        output_format="LABEL_CXCYWH")
                                    ])


train_loader = dataloaders.get(dataset=trainset, dataloader_params={
    "shuffle": True,
    "batch_size": 32,
    "drop_last": False,
    "pin_memory": True,
    "collate_fn": CrowdDetectionCollateFN(),
    "worker_init_fn": worker_init_reset_seed,
    "min_samples": 512
})

valid_loader = dataloaders.get(dataset=valset, dataloader_params={
    "shuffle": False,
    "batch_size": 32,
    "num_workers": 2,
    "drop_last": False,
    "pin_memory": True,
    "collate_fn": CrowdDetectionCollateFN(),
    "worker_init_fn": worker_init_reset_seed
})


train_params = {
    'silent_mode': False,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": params['initial_lr'],
    "resume": params['resume'],
    "lr_mode": params['lr_mode'],
    "cosine_final_lr_ratio": params['cosine_final_lr_ratio'],
    "optimizer": params['optimizer'],
    "optimizer_params": params['optimizer_params'],
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": params['ema_params'],
    "max_epochs": params['max_epochs'],
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(CLASSES),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.3,
            top_k_predictions=300,
            num_cls=len(CLASSES),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        ),
        DetectionMetrics_050_3Main_Class(
            score_thres=0.3,
            top_k_predictions=300,
            num_cls=len(CLASSES),
            normalize_targets=True, 
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        ),
        DetectionMetrics_050_5Classes(
            score_thres=0.3,
            top_k_predictions=300,
            num_cls=len(CLASSES),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        ),
        DetectionMetrics_050_095(
            score_thres=0.3,
            top_k_predictions=300,
            num_cls=len(CLASSES),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )

    ],
    "metric_to_watch": 'mAP@0.50'
}

model = models.get(
    MODEL_ARCH, 
    num_classes=len(CLASSES), 
    pretrained_weights="coco"
)


# 
# print(train_data)
trainer.train(
    model=model, 
    training_params=train_params, 
    train_loader=train_loader, 
    valid_loader=valid_loader
)
