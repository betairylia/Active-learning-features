from .nets import *

net_dict =\
{
    'mlp': MLPFactory,
    'simple-cnn': CNNFactory,
    'resnet-cifar': ResNetCIFARFactory,
    'resnet18-imagenet': ResNet18ImageNetPretrainedFactory,
    'resnet50-imagenet': ResNet50ImageNetPretrainedFactory,
    'wide-resnet50-imagenet': WideResNet50ImageNetPretrainedFactory,
    'lenet-5': LeNetFactory,
}
