import timm


def build_model(config):
    model = timm.create_model(
        config.model_name,
        pretrained=True,
        num_classes=config.num_classes
    )

    return model