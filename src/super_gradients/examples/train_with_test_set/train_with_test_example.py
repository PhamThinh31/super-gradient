from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.dataloaders.dataloaders import cifar10_train, cifar10_val


trainer = Trainer(experiment_name="demo-train-with-test")
model = models.get(Models.RESNET18, num_classes=10)
training_params = {
    "max_epochs": 20,
    "lr_updates": [5, 10, 15],
    "lr_decay_factor": 0.1,
    "lr_mode": "step",
    "initial_lr": 0.1,
    "loss": "CrossEntropyLoss",
    "optimizer": "SGD",
    "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
    "train_metrics_list": [Accuracy(), Top5()],
    "valid_metrics_list": [Accuracy(), Top5()],
    "metric_to_watch": "Accuracy",
}

trainer.train(
    model=model,
    training_params=training_params,
    train_loader=cifar10_train(),
    valid_loader=cifar10_val(),
    test_loaders={
        "Copy of train": cifar10_val(),
        "Copy of valid": cifar10_val(),
    },  # For the sake of the example, we are testing on the training and validation set.
)
