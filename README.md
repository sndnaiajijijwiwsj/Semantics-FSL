# Multi-Attention Based Visual-Semantic Interaction for Few-Shot Learning

The code repository for "Multi-Attention Based Visual-Semantic Interaction for Few-Shot Learning"


## Requirements

The following packages are required to run the scripts:

- PyTorch-1.1 and torchvision

- Dataset: please download the dataset and put images into the folder data/images

- Download MiniImageNet and place it under `data/miniimagenet/`.

## Running the code

For example, to train the 5-shot 5-way CombinedProtoNet model with ResNet-12 backbone on MiniImageNet:

    $ python train_fsl.py --model_class CombinedProtoNet  --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15

For further arguments please take a look at `model/utils.py`.
