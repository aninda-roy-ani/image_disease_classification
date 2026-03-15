import timm

"""
def build_model(config):
    model = timm.create_model(
        config.model_name,
        pretrained=True,
        num_classes=config.num_classes
    )

    return model
"""

import torch.nn as nn
from src.activations import RepAct_Softmax, RepAct_Origin, RepAct_BN


def replace_gelu_with_repact(module):
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            print("FOUND")
            setattr(module, name, RepAct_BN())
        else:
            replace_gelu_with_repact(child)


def build_model(config, use_repact=False):
    model = timm.create_model(
        config.model_name,
        pretrained=True,
        num_classes=config.num_classes
    )

    if use_repact:
        replace_gelu_with_repact(model)

    return model
