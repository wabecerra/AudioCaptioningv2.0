import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.models.feature_extraction import create_feature_extractor
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

def get_audio_feature_vector(
    model: nn.Module,
    img_tensors: torch.Tensor,
    layer_dict: dict
) -> torch.Tensor:
    """
    Get feature vectors for the given audio (passed in as a spectrogram image) by returning the output
    of the selected model at an intermediate layer.

    For more info on create_feature_extractor, see the Torchvision documentation here:
    http://pytorch.org/vision/main/generated/torchvision.models.feature_extraction.create_feature_extractor.html

    Arguments
    -----
    model - A PyTorch nn.Module object
    img_tensor - A batch of images stored as PyTorch tensors with dimensions (n, c, h, w)
    layer_dict - A dictionary containing information about the desired intermediate layer output
        Ex. {'avgpool': 'features'} to obtain the weights at ResNet's avgpool layer stored with the key "features"

    Outputs
    -----
    feat_vec - A list of feature vectors for each of the requested layers as tuples in [("name", tensor)] format

    """
    extractor = create_feature_extractor(model, layer_dict)
    img_tensors = img_tensors.unsqueeze(0)
    extractor = extractor.double()
    out = extractor(img_tensors)
    features = out["features"]
    return features.view(features.size(0), -1)

def get_vit_feature_vector(
    model: nn.Module,
    device: torch.device,
    img_tensors: torch.Tensor,
    layer_dict: dict = {"encoder.layers.encoder_layer_11.mlp": "features"}
) -> torch.Tensor:
    """
    Get feature vectors for the given audio (passed in as a spectrogram image) by returning the output
    of the selected model at an intermediate layer.

    For more info on create_feature_extractor, see the Torchvision documentation here:
    http://pytorch.org/vision/main/generated/torchvision.models.feature_extraction.create_feature_extractor.html

    Arguments
    -----
    model - A PyTorch nn.Module object
    img_tensor - A batch of images stored as PyTorch tensors with dimensions (n, c, h, w)
    layer_dict - A dictionary containing information about the desired intermediate layer output
        Ex. {'avgpool': 'features'} to obtain the weights at ResNet's avgpool layer stored with the key "features"

    Outputs
    -----
    feat_vec - A list of feature vectors for each of the requested layers as tuples in [("name", tensor)] format

    """
    extractor = create_feature_extractor(model.to(device), layer_dict)
    img_tensors = img_tensors.unsqueeze(0).to(device)
    extractor = extractor.double()
    out = extractor(img_tensors)
    features = out["features"]  # shape: (batch, seq_len, hidden_dim)
    # Typically the [CLS] token is at index 0
    return features[:, 0, :]


def init_layer(layer: nn.Module) -> None:
    """
    Initialize a Linear or Convolutional layer using Xavier uniform.
    """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn: nn.BatchNorm2d) -> None:
    """
    Initialize a BatchNorm2d layer with bias=0 and weight=1.
    """
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    """
    A simple two-convolution block with optional average or max pooling.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self) -> None:
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x: torch.Tensor, pool_size=(2, 2), pool_type='avg') -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = (F.avg_pool2d(x, kernel_size=pool_size) +
                 F.max_pool2d(x, kernel_size=pool_size))
        else:
            raise ValueError("Unknown pool_type: {}".format(pool_type))
        return x


class Cnn14(nn.Module):
    """
    A CNN architecture from PANN (Pretrained Audio Neural Networks).
    """
    def __init__(self, classes_num: int = 527):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.init_weight()

    def init_weight(self) -> None:
        init_bn(self.bn0)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_data: (batch_size, time_length) or (batch_size, 1, freq, time)
                       For PANN, typically (batch_size, 64, time).
        Returns:
            A (batch_size, 2048) tensor of extracted features.
        """
        x = input_data.unsqueeze(1).float()  # (batch_size, 1, freq, time)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # Now shape is (batch_size, 2048, freq=1, time=?)
        x = torch.mean(x, dim=3)  # (batch_size, 2048, freq=1)
        x1, _ = torch.max(x, dim=2)  # (batch_size, 2048)
        x2 = torch.mean(x, dim=2)    # (batch_size, 2048)
        return x1 + x2  # (batch_size, 2048)