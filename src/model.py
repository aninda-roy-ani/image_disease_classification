import timm
import torch.nn as nn


def build_model(config):
    model = timm.create_model(
        config.model_name,
        pretrained=True
    )

    in_features = model.head.in_features
    model.head = nn.Linear(in_features, config.num_classes)

    return model