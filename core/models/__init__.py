from core.models.segnet import SegNet
from core.models.unet import UNet
from core.models.deconvnet import DeconvNet


def load_empty_model(architecture, n_classes):
    models = {
        'segnet': SegNet,
        'unet': UNet,
        'deconvnet': DeconvNet
    }

    ModelClass = models[architecture]
    model = ModelClass(n_channels=1, n_classes=n_classes)

    return model
