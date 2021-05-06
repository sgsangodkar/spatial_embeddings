from models.unet import ResUNet
from models.erfnet import BranchedERFNet


def get_model(name, model_opts):
    if name == "unet": 
        return ResUNet(**model_opts)
    if name == "erfnet":
        return BranchedERFNet(**model_opts) 
    else:
        raise RuntimeError("Model {} not available".format(name))


if __name__ == '__main__':
    options = dict(decoder_channels = [3, 1])
    model = get_model('erfnet', options)
    print(model)