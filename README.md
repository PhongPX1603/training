# training

## Folder Contructor
```
pytorch-template/
│
├── train.py - main script to start training
├── test.py - evaluation of trained model
│
├── utils.py
│
├── configs
│   ├── mnist.yaml
│   ├── cifar10.yaml
│   └── ...
│
├── metrics
│   ├── metric_funcs.py
│   └── metrics.py
│
├── model
│   ├── mnist_model.py
│   ├── cifar_model.py
│   └── ...
│
├── trainer
│   ├── early_stopping.py
│   └── trainer.py
│
└── ...
```

## config flie format

Config files are in ```.yaml``` format
```
data:
  train:
    module: torch.utils.data
    class: DataLoader
    DataLoader:
      dataset:
        module: torchvision.datasets
        class: CIFAR10
        CIFAR10:
          root: '''datasets/cifar'''
          train: True
          download: True
          transform: transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
      batch_size: 64
      shuffle: True
      num_workers: 2

  valid:
    module: torch.utils.data
    class: DataLoader
    DataLoader:
      dataset:
        module: torchvision.datasets
        class: CIFAR10
        CIFAR10:
          root: '''datasets/cifar'''
          train: False
          download: True
          transform: transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
      batch_size: 64
      shuffle: True
      num_workers: 2

model:
  module: models.mobilenet_v2
  class: MobileNetV2
  MobileNetV2:
    num_classes: 10

loss:
  module: torch.nn
  class: CrossEntropyLoss

optimizer:
  module: torch.optim
  class: Adam
  Adam:
    params: config['model'].parameters()
    lr: 0.001
    amsgrad: True

lr_scheduler:
  module: torch.optim.lr_scheduler
  class: ReduceLROnPlateau
  ReduceLROnPlateau:
    optimizer: config['optimizer']
    mode: '''min'''
    factor: 0.1
    patience: 3
    verbose: True

metrics:
  module: metrics.metrics
  class: Metric
  Metric:
    metric:
      accuracy:
        module: metrics.metric_funcs
        class: Accuracy
      loss:
        module: torch.nn
        class: CrossEntropyLoss
    output_transform: 'lambda x: x'

extralibs:
  torch: torch
  torchvision: torchvision
  transforms: torchvision.transforms
```

## Training
```
python train.py --config-path (str-config path) --num-epochs (int-number epochs) --project-name (str-name of project) --device (str: cpu, cuda) --save-weight-dir (str-direction of folder to save weights)
```

### tensorboard
```
tensorboard --logdir (direction of tensorboard file or folder)
```
