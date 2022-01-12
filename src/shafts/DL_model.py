import os
import math
from .DL_module import *


# ************************* Backbones *************************
class ResNetBackbone(nn.Module):
    def __init__(self, input_channels: int, input_size: int, in_plane=64, num_block=2):
        """Initializer for the backbone of ResNet.

        Parameters
        ----------

        input_channels : int
            Number of input channels.
        input_size : int
            Size of input patches.
        in_plane : int
            Number of output channels after the initial convolutional layer.
            The default is `64`.
        num_block : int
            Number of ResBlocks in each ResLayer.
            The default is `2`.

        """
        super(ResNetBackbone, self).__init__()

        self.in_plane = in_plane
        #self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_plane)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = None
        if input_size in [120]:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=7, stride=2, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        elif input_size in [60]:
            # self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=5, stride=2, padding=1, bias=False)
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=7, stride=2, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        else:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=3, stride=1, padding=1, bias=False)
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 1)

        res_layer = [self._make_layer(in_plane, blocks=num_block)]

        if num_reduce == 1:
            in_plane = in_plane * 2
            res_layer.append(self._make_layer(in_plane, blocks=num_block))

        for i in range(0, num_reduce):
            num_plane = int(in_plane * math.pow(2, i+1))
            res_layer.append(self._make_layer(num_plane, blocks=num_block, stride=2))

        self.res_layer = nn.Sequential(*res_layer)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, num_plane: int, blocks: int, stride=1):
        """Generator for ResLayers.

        Parameters
        ----------

        num_plane : int
            Number of output channels.
        blocks : int
            Number of ResBlocks in the ResLayer.
        stride : int
            Stride for the convolution operation used in the initial ResBlock.
            The default is `1`.

        """
        downsample = None
        if (stride != 1) or (self.in_plane != num_plane):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_plane, out_channels=num_plane, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_plane)
            )
        layers = [BasicResBlock(self.in_plane, num_plane, stride, downsample=downsample)]
        self.in_plane = num_plane
        for _ in range(1, blocks):
            layers.append(BasicResBlock(self.in_plane, num_plane))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.maxpool is not None:
            x = self.maxpool(x)

        out = self.res_layer(x)

        return out


class SENetBackbone(nn.Module):
    def __init__(self, input_channels: int, input_size: int, in_plane=64, num_block=2):
        """Initializer for the backbone of SENet.

        Parameters
        ----------

        input_channels : int
            Number of input channels.
        input_size : int
            Size of input patches.
        in_plane : int
            Number of output channels after the initial convolutional layer.
            The default is `64`.
        num_block : int
            Number of SEBlocks in each SELayer.
            The default is `2`.

        """
        super(SENetBackbone, self).__init__()
        self.in_plane = in_plane
        #self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_plane)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = None
        if input_size in [120]:
            # self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=5, stride=2, padding=1, bias=False)
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=7, stride=2, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        elif input_size in [60]:
            # self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=5, stride=2, padding=1, bias=False)
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=7, stride=2, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        else:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=3, stride=1, padding=1, bias=False)
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 1)

        res_layer = [self._make_layer(in_plane, blocks=num_block)]

        if num_reduce == 1:
            in_plane = in_plane * 2
            res_layer.append(self._make_layer(in_plane, blocks=num_block))

        for i in range(0, num_reduce):
            num_plane = int(in_plane * math.pow(2, i+1))
            res_layer.append(self._make_layer(num_plane, blocks=num_block, stride=2))

        self.res_layer = nn.Sequential(*res_layer)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, num_plane: int, blocks: int, stride=1):
        """Generator for SELayers.

        Parameters
        ----------

        num_plane : int
            Number of output channels.
        blocks : int
            Number of SEBlocks in the SELayer.
        stride : int
            Stride for the convolution operation used in the initial SEBlock.
            The default is `1`.

        """
        downsample = None
        if (stride != 1) or (self.in_plane != num_plane):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_plane, out_channels=num_plane, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_plane)
            )
        layers = [SEResBlock(self.in_plane, num_plane, stride, downsample=downsample)]
        self.in_plane = num_plane
        for _ in range(1, blocks):
            layers.append(SEResBlock(self.in_plane, num_plane))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.maxpool is not None:
            x = self.maxpool(x)

        out = self.res_layer(x)

        return out


class CBAMBackbone(nn.Module):
    def __init__(self, input_channels: int, input_size: int, in_plane=64, num_block=2):
        """Initializer for the backbone of CBAM.

        Parameters
        ----------

        input_channels : int
            Number of input channels.
        input_size : int
            Size of input patches.
        in_plane : int
            Number of output channels after the initial convolutional layer.
            The default is `64`.
        num_block : int
            Number of CBAMBlocks in each CBAMLayer.
            The default is `2`.

        """
        super(CBAMBackbone, self).__init__()
        self.in_plane = in_plane
        #self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_plane)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = None
        if input_size in [120]:
            # self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=7, stride=1, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        elif input_size in [60]:
            # self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=7, stride=1, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 2)
        else:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=3, stride=1, padding=1, bias=False)
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 1)

        res_layer = [self._make_layer(in_plane, blocks=num_block)]

        if num_reduce == 1:
            in_plane = in_plane * 2
            res_layer.append(self._make_layer(in_plane, blocks=num_block))

        for i in range(0, num_reduce):
            num_plane = int(in_plane * math.pow(2, i+1))
            res_layer.append(self._make_layer(num_plane, blocks=num_block, stride=2))

        self.res_layer = nn.Sequential(*res_layer)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, num_plane: int, blocks: int, stride=1):
        """Generator for CBAMLayers.

        Parameters
        ----------

        num_plane : int
            Number of output channels.
        blocks : int
            Number of CBAMBlocks in the CBAMLayer.
        stride : int
            Stride for the convolution operation used in the initial CBAMBlock.
            The default is `1`.

        """
        downsample = None
        if (stride != 1) or (self.in_plane != num_plane):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_plane, out_channels=num_plane, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_plane)
            )
        layers = [CBAMResBlock(self.in_plane, num_plane, stride, downsample=downsample)]
        self.in_plane = num_plane
        for _ in range(1, blocks):
            layers.append(CBAMResBlock(self.in_plane, num_plane))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.maxpool is not None:
            x = self.maxpool(x)

        out = self.res_layer(x)

        return out


# ************************* CNNs *************************
class BuildingNet(nn.Module):
    def __init__(self, input_channels: int, input_size: int, backbone: str, in_plane=64, num_block=2, log_scale=False, activation="relu", cuda_used=True, **kwargs):
        """Initializer for CNN trained by Single Task Learning (STL) for 3D building information extraction.

        Parameters
        ----------

        input_channels : int
            Number of input channels.
        input_size : int
            Size of input patches.
        backbone : str
            Name of CNN's backbone for feature extraction.
            It can be chosen from: `ResNet`, `SENet`, `CBAM`.
        in_plane : int
            Number of output channels after the initial convolutional layer.
            The default is `64`.
        num_block : int
            Number of CNNBlocks in each CNNLayer.
            The default is `2`.
        log_scale : boolean
            A flag which controls whether log-transformation is used for output.
            The default is `False`.
        activation : str
            Activation function for model output.
            It can be chosen from: `relu`, `sigmoid`. The default is `relu`.
        cuda_used : boolean
            A flag which controls whether CUDA is used for inference.
            The default is `False`.

        """
        super(BuildingNet, self).__init__()
        self.cuda_used = cuda_used
        if backbone == "ResNet":
            self.features = ResNetBackbone(input_channels, input_size, in_plane=in_plane, num_block=num_block)
        elif backbone == "SENet":
            self.features = SENetBackbone(input_channels, input_size, in_plane=in_plane, num_block=num_block)
        elif backbone == "CBAM":
            self.features = CBAMBackbone(input_channels, input_size, in_plane=in_plane, num_block=num_block)
        else:
            raise NotImplementedError("Unknown backbone!")

        self.log_scale = log_scale
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)

        if input_size in [120]:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        elif input_size in [60]:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 2)
        else:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 1)

        if num_reduce == 1:
            num_plane = int(in_plane * math.pow(2, num_reduce + 1))
        else:
            num_plane = int(in_plane * math.pow(2, num_reduce))

        self.fc = nn.Linear(num_plane, int(num_plane / 2))
        self.bn_out = nn.BatchNorm1d(int(num_plane / 2))
        self.fc_out = nn.Linear(int(num_plane / 2), 1)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # load pretrained weights
        if "trained_record" in kwargs.keys() and kwargs["trained_record"] is not None:
            self.load_pretrained_model(trained_record=kwargs["trained_record"])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.bn_out(x)
        x = self.relu(x)
        # x = self.drop_out(x)
        x = self.fc_out(x)

        # ---For BuildingFootprint prediction, we set log_scale=False by default
        if not self.log_scale:
            x = self.activation(x)

        return x

    def load_pretrained_model(self, trained_record: str):
        """Load pretrained models.

        Parameters
        ----------

        trained_record : str
            Path to the pretrained model file.
        
        """
        if self.cuda_used:
            checkpoint = torch.load(trained_record)
        else:
            checkpoint = torch.load(trained_record, map_location=torch.device('cpu'))
        pretrained_dict = checkpoint["state_dict"]
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrained_dict.items():
            if k in state_dict and (v.size() == state_dict[k].size()):
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def get_feature(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        return x


class BuildingNet_aux(nn.Module):
    def __init__(self, input_channels: int, input_size: int, aux_input_size: int, backbone: str, in_plane=64, num_aux=1, num_block=2, log_scale=False, activation="relu", cuda_used=True, **kwargs):
        """Initializer for CNN trained by Single Task Learning (STL) for 3D building information extraction with auxiliary input information.

        Parameters
        ----------

        input_channels : int
            Number of input channels.
        input_size : int
            Size of Sentinel's input patches.
        aux_input_size : int
            Size of auxiliary input patches.
        backbone : str
            Name of CNN's backbone for feature extraction.
            It can be chosen from: `ResNet`, `SENet`, `CBAM`.
        in_plane : int
            Number of output channels after the initial convolutional layer.
            The default is `64`.
        num_aux : int
            Number of auxiliary variables.
            The default is `1`.
        num_block : int
            Number of CNNBlocks in each CNNLayer.
            The default is `2`.
        log_scale : boolean
            A flag which controls whether log-transformation is used for output.
            The default is `False`.
        activation : str
            Activation function for model output.
            It can be chosen from: `relu`, `sigmoid`. The default is `relu`.
        cuda_used : boolean
            A flag which controls whether CUDA is used for inference.
            The default is `False`.

        """
        super(BuildingNet_aux, self).__init__()
        self.cuda_used = cuda_used
        # aux_in_plane = int(in_plane)
        aux_in_plane = int(in_plane / 4.0)
        if backbone == "ResNet":
            self.features = ResNetBackbone(input_channels, input_size, in_plane=in_plane, num_block=num_block)
            self.aux_features = ResNetBackbone(num_aux, aux_input_size, in_plane=aux_in_plane, num_block=num_block)
        elif backbone == "SENet":
            self.features = SENetBackbone(input_channels, input_size, in_plane=in_plane, num_block=num_block)
            self.aux_features = SENetBackbone(num_aux, aux_input_size, in_plane=aux_in_plane, num_block=num_block)
        elif backbone == "CBAM":
            self.features = CBAMBackbone(input_channels, input_size, in_plane=in_plane, num_block=num_block)
            self.aux_features = CBAMBackbone(num_aux, aux_input_size, in_plane=aux_in_plane, num_block=num_block)
        else:
            raise NotImplementedError("Unknown backbone!")

        self.log_scale = log_scale
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)

        if input_size in [120]:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
            aux_num_reduce = int(math.floor(math.log(aux_input_size / 2, 2)) - 3)
        elif input_size in [60]:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 2)
            aux_num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 2)
        else:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 1)
            aux_num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 1)

        if num_reduce == 1:
            num_plane = int(in_plane * math.pow(2, num_reduce + 1))
        else:
            num_plane = int(in_plane * math.pow(2, num_reduce))
        
        if aux_num_reduce == 1:
            aux_num_plane = int(aux_in_plane * math.pow(2, aux_num_reduce + 1))
        else:
            aux_num_plane = int(aux_in_plane * math.pow(2, aux_num_reduce))
        
        num_plane = num_plane + aux_num_plane

        self.fc = nn.Linear(num_plane, int(num_plane / 2))
        self.bn_out = nn.BatchNorm1d(int(num_plane / 2))
        self.fc_out = nn.Linear(int(num_plane / 2), 1)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # load pretrained weights
        if "trained_record" in kwargs.keys() and kwargs["trained_record"] is not None:
            self.load_pretrained_model(trained_record=kwargs["trained_record"])

    def forward(self, x, aux):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        aux = self.aux_features(aux)
        aux = self.avgpool(aux)
        aux = torch.flatten(aux, start_dim=1)

        x = torch.cat((x, aux), dim=1)

        x = self.fc(x)
        x = self.bn_out(x)
        x = self.relu(x)
        x = self.fc_out(x)

        # ---For BuildingFootprint prediction, we set log_scale=False by default
        if not self.log_scale:
            x = self.activation(x)

        return x

    def load_pretrained_model(self, trained_record: str):
        """Load pretrained models.

        Parameters
        ----------

        trained_record : str
            Path to the pretrained model file.
        
        """
        if self.cuda_used:
            checkpoint = torch.load(trained_record)
        else:
            checkpoint = torch.load(trained_record, map_location=torch.device('cpu'))
        pretrained_dict = checkpoint["state_dict"]
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrained_dict.items():
            if k in state_dict and (v.size() == state_dict[k].size()):
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def get_feature(self, x, aux):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        aux = self.aux_features(aux)
        aux = self.avgpool(aux)
        aux = torch.flatten(aux, start_dim=1)

        x = torch.cat((x, aux), dim=1)

        return x


# ************************* CNNs in their Multi-Task Learning version *************************
class BuildingNetMTL(nn.Module):
    def __init__(self, input_channels: int, input_size: int, backbone: str, in_plane=64, num_block=2, crossed=False, log_scale=False, cuda_used=True, **kwargs):
        """Initializer for CNN trained by Multi-Task Learning (MTL) for 3D building information extraction.

        Parameters
        ----------

        input_channels : int
            Number of input channels.
        input_size : int
            Size of input patches.
        backbone : str
            Name of CNN's backbone for feature extraction.
            It can be chosen from: `ResNet`, `SENet`, `CBAM`.
        in_plane : int
            Number of output channels after the initial convolutional layer.
            The default is `64`.
        num_block : int
            Number of CNNBlocks in each CNNLayer.
            The default is `2`.
        crossed : boolean
            A flag which controls whether a link is to be added between last fully-connected layers of building footprint and height prediction.
            The default is `False`.
        log_scale : boolean
            A flag which controls whether log-transformation is used for output.
            The default is `False`.
        cuda_used : boolean
            A flag which controls whether CUDA is used for inference.
            The default is `False`.

        """
        super(BuildingNetMTL, self).__init__()
        self.crossed = crossed
        self.cuda_used = cuda_used
        if backbone == "ResNet":
            self.features = ResNetBackbone(input_channels, input_size, in_plane=in_plane, num_block=num_block)
        elif backbone == "SENet":
            self.features = SENetBackbone(input_channels, input_size, in_plane=in_plane, num_block=num_block)
        elif backbone == "CBAM":
            self.features = CBAMBackbone(input_channels, input_size, in_plane=in_plane, num_block=num_block)
        else:
            raise NotImplementedError("Unknown backbone!")

        self.log_scale = log_scale
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        if input_size in [120]:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        elif input_size in [60]:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 2)
        else:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 1)

        if num_reduce == 1:
            num_plane = int(in_plane * math.pow(2, num_reduce + 1))
        else:
            num_plane = int(in_plane * math.pow(2, num_reduce))

        self.fc_height = nn.Linear(num_plane, int(num_plane / 2))
        self.bn_out_height = nn.BatchNorm1d(int(num_plane / 2))
        self.fc_out_height = nn.Linear(int(num_plane / 2), 1)

        self.fc_footprint = nn.Linear(num_plane, int(num_plane / 2))
        self.bn_out_footprint = nn.BatchNorm1d(int(num_plane / 2))
        self.fc_out_footprint = nn.Linear(int(num_plane / 2), 1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # load pretrained weights
        if "trained_record" in kwargs.keys() and kwargs["trained_record"] is not None:
            self.load_pretrained_model(trained_record=kwargs["trained_record"])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        feature_share = torch.flatten(x, start_dim=1)

        feature_fc1_footprint = self.fc_footprint(feature_share)
        feature_fc1_footprint = self.bn_out_footprint(feature_fc1_footprint)
        feature_fc1_footprint = self.relu(feature_fc1_footprint)

        footprint = self.fc_out_footprint(feature_fc1_footprint)
        footprint = self.sigmoid(footprint)

        feature_fc1_height = self.fc_height(feature_share)
        feature_fc1_height = self.bn_out_height(feature_fc1_height)
        feature_fc1_height = self.relu(feature_fc1_height)

        if self.crossed:
            feature_fc1_height = feature_fc1_height * feature_fc1_footprint

        height = self.fc_out_height(feature_fc1_height)

        if not self.log_scale:
            height = self.relu(height)

        return footprint, height

    def load_pretrained_model(self, trained_record: str):
        """Load pretrained models.

        Parameters
        ----------

        trained_record : str
            Path to the pretrained model file.
        
        """
        if self.cuda_used:
            checkpoint = torch.load(trained_record)
        else:
            checkpoint = torch.load(trained_record, map_location=torch.device('cpu'))
        pretrained_dict = checkpoint["state_dict"]
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrained_dict.items():
            if (k in state_dict) and (v.size() == state_dict[k].size()):
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def get_feature(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        return x


class BuildingNetMTL_aux(nn.Module):
    def __init__(self, input_channels: int, input_size: int, aux_input_size: int, backbone: str, in_plane=64, num_aux=1, num_block=2, crossed=False, log_scale=False, cuda_used=True, **kwargs):
        """Initializer for CNN trained by Multi-Task Learning (MTL) for 3D building information extraction with auxiliary input information.

        Parameters
        ----------

        input_channels : int
            Number of input channels.
        input_size : int
            Size of Sentinel's input patches.
        aux_input_size : int
            Size of auxiliary input patches.
        backbone : str
            Name of CNN's backbone for feature extraction.
            It can be chosen from: `ResNet`, `SENet`, `CBAM`.
        in_plane : int
            Number of output channels after the initial convolutional layer.
            The default is `64`.
        num_aux : int
            Number of auxiliary variables.
            The default is `1`.
        num_block : int
            Number of CNNBlocks in each CNNLayer.
            The default is `2`.
        crossed : boolean
            A flag which controls whether a link is to be added between last fully-connected layers of building footprint and height prediction.
            The default is `False`.
        log_scale : boolean
            A flag which controls whether log-transformation is used for output.
            The default is `False`.
        cuda_used : boolean
            A flag which controls whether CUDA is used for inference.
            The default is `False`.

        """
        super(BuildingNetMTL_aux, self).__init__()
        self.crossed = crossed
        self.cuda_used = cuda_used
        # aux_in_plane = int(in_plane)
        aux_in_plane = int(in_plane / 4.0)
        if backbone == "ResNet":
            self.features = ResNetBackbone(input_channels, input_size, in_plane=in_plane, num_block=num_block)
            self.aux_features = ResNetBackbone(num_aux, aux_input_size, in_plane=aux_in_plane, num_block=num_block)
        elif backbone == "SENet":
            self.features = SENetBackbone(input_channels, input_size, in_plane=in_plane, num_block=num_block)
            self.aux_features = SENetBackbone(num_aux, aux_input_size, in_plane=aux_in_plane, num_block=num_block)
        elif backbone == "CBAM":
            self.features = CBAMBackbone(input_channels, input_size, in_plane=in_plane, num_block=num_block)
            self.aux_features = CBAMBackbone(num_aux, aux_input_size, in_plane=aux_in_plane, num_block=num_block)
        else:
            raise NotImplementedError("Unknown backbone!")

        self.log_scale = log_scale
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        if input_size in [120]:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
            aux_num_reduce = int(math.floor(math.log(aux_input_size / 2, 2)) - 3)
        elif input_size in [60]:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 2)
            aux_num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 2)
        else:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 1)
            aux_num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 1)

        if num_reduce == 1:
            num_plane = int(in_plane * math.pow(2, num_reduce + 1))
        else:
            num_plane = int(in_plane * math.pow(2, num_reduce))

        if aux_num_reduce == 1:
            aux_num_plane = int(aux_in_plane * math.pow(2, aux_num_reduce + 1))
        else:
            aux_num_plane = int(aux_in_plane * math.pow(2, aux_num_reduce))
        
        num_plane = num_plane + aux_num_plane

        self.fc_height = nn.Linear(num_plane, int(num_plane / 2))
        self.bn_out_height = nn.BatchNorm1d(int(num_plane / 2))
        self.fc_out_height = nn.Linear(int(num_plane / 2), 1)

        self.fc_footprint = nn.Linear(num_plane, int(num_plane / 2))
        self.bn_out_footprint = nn.BatchNorm1d(int(num_plane / 2))
        self.fc_out_footprint = nn.Linear(int(num_plane / 2), 1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # load pretrained weights
        if "trained_record" in kwargs.keys() and kwargs["trained_record"] is not None:
            self.load_pretrained_model(trained_record=kwargs["trained_record"])

    def forward(self, x, aux):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        aux = self.aux_features(aux)
        aux = self.avgpool(aux)
        aux = torch.flatten(aux, start_dim=1)

        feature_share = torch.cat((x, aux), dim=1)

        feature_fc1_footprint = self.fc_footprint(feature_share)
        feature_fc1_footprint = self.bn_out_footprint(feature_fc1_footprint)
        feature_fc1_footprint = self.relu(feature_fc1_footprint)

        footprint = self.fc_out_footprint(feature_fc1_footprint)
        footprint = self.sigmoid(footprint)

        feature_fc1_height = self.fc_height(feature_share)
        feature_fc1_height = self.bn_out_height(feature_fc1_height)
        feature_fc1_height = self.relu(feature_fc1_height)

        if self.crossed:
            feature_fc1_height = feature_fc1_height * feature_fc1_footprint

        height = self.fc_out_height(feature_fc1_height)

        if not self.log_scale:
            height = self.relu(height)

        return footprint, height

    def load_pretrained_model(self, trained_record: str):
        """Load pretrained models.

        Parameters
        ----------

        trained_record : str
            Path to the pretrained model file.
        
        """
        if self.cuda_used:
            checkpoint = torch.load(trained_record)
        else:
            checkpoint = torch.load(trained_record, map_location=torch.device('cpu'))
        pretrained_dict = checkpoint["state_dict"]
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrained_dict.items():
            if (k in state_dict) and (v.size() == state_dict[k].size()):
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def get_feature(self, x, aux):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        aux = self.aux_features(aux)
        aux = self.avgpool(aux)
        aux = torch.flatten(aux, start_dim=1)

        x = torch.cat((x, aux), dim=1)

        return x


# ************************* CNN instances using ResNet as backbones *************************
def model_ResNet(input_channels: int, input_size: int, in_plane: int, num_block: int, log_scale=False, activation="relu", cuda_used=False, **kwargs) -> BuildingNet:
    """Prepare ResNet-STL for 3D building information extraction.

    Parameters
    ----------

    input_channels : int
        Number of input channels.
    input_size : int
        Size of input patches.
    in_plane : int
        Number of output channels after the initial convolutional layer.
    num_block : int
        Number of ResBlocks in each ResLayer.
    log_scale : boolean
        A flag which controls whether log-transformation is used for output.
        The default is `False`.
    activation : str
        Activation function for model output.
        It can be chosen from: `relu`, `sigmoid`. The default is `relu`.
    cuda_used : boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.

    """
    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
    else:
        trained_record = None

    model = BuildingNet(input_channels=input_channels, input_size=input_size, backbone="ResNet", in_plane=in_plane,
                        num_block=num_block, log_scale=log_scale, activation=activation, cuda_used=cuda_used, trained_record=trained_record)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter of ResNet: ", total_num, " Trainable parameter of ResNet: ", trainable_num)
    return model


def model_ResNet_aux(input_channels: int, input_size: int, aux_input_size: int, in_plane: int, num_block: int, num_aux=1, log_scale=False, activation="relu", cuda_used=False, **kwargs) -> BuildingNet_aux:
    """Prepare ResNet-STL using auxiliary input information for 3D building information extraction.

    Parameters
    ----------

    input_channels : int
        Number of input channels.
    input_size : int
        Size of input patches.
    aux_input_size : int
        Size of auxiliary input patches.
    in_plane : int
        Number of output channels after the initial convolutional layer.
    num_block : int
        Number of ResBlocks in each ResLayer.
    num_aux : int
        Number of auxiliary variables.
        The default is `1`.
    log_scale : boolean
        A flag which controls whether log-transformation is used for output.
        The default is `False`.
    activation : str
        Activation function for model output.
        It can be chosen from: `relu`, `sigmoid`. The default is `relu`.
    cuda_used : boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.

    """
    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
    else:
        trained_record = None

    model = BuildingNet_aux(input_channels=input_channels, input_size=input_size, aux_input_size=aux_input_size, backbone="ResNet", in_plane=in_plane,
                                num_aux=num_aux, num_block=num_block, log_scale=log_scale, activation=activation, 
                                cuda_used=cuda_used, trained_record=trained_record)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter of ResNet: ", total_num, " Trainable parameter of ResNet: ", trainable_num)
    return model


def model_ResNetMTL(input_channels: int, input_size: int, in_plane: int, num_block: int, crossed=False, log_scale=False, cuda_used=False, **kwargs) -> BuildingNetMTL:
    """Prepare ResNet-MTL using auxiliary input information for 3D building information extraction.

    Parameters
    ----------

    input_channels : int
        Number of input channels.
    input_size : int
        Size of input patches.
    in_plane : int
        Number of output channels after the initial convolutional layer.
    num_block : int
        Number of ResBlocks in each ResLayer.
    crossed : boolean
        A flag which controls whether a link is to be added between last fully-connected layers of building footprint and height prediction.
        The default is `False`.
    log_scale : boolean
        A flag which controls whether log-transformation is used for output.
        The default is `False`.
    cuda_used : boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.

    """
    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
    else:
        trained_record = None

    model = BuildingNetMTL(input_channels=input_channels, input_size=input_size, backbone="ResNet", in_plane=in_plane,
                           num_block=num_block, crossed=crossed, log_scale=log_scale,
                           cuda_used=cuda_used, trained_record=trained_record)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter of ResNetMTL: ", total_num, " Trainable parameter of ResNetMTL: ", trainable_num)
    return model


def model_ResNetMTL_aux(input_channels: int, input_size: int, aux_input_size: int, in_plane: int, num_block: int, num_aux=1, crossed=False, log_scale=False, cuda_used=False, **kwargs) -> BuildingNetMTL_aux:
    """Prepare ResNet-MTL using auxiliary input information for 3D building information extraction.

    Parameters
    ----------

    input_channels : int
        Number of input channels.
    input_size : int
        Size of input patches.
    aux_input_size : int
        Size of auxiliary input patches.
    in_plane : int
        Number of output channels after the initial convolutional layer.
    num_block : int
        Number of ResBlocks in each ResLayer.
    num_aux : int
        Number of auxiliary variables.
        The default is `1`.
    crossed : boolean
        A flag which controls whether a link is to be added between last fully-connected layers of building footprint and height prediction.
        The default is `False`.
    log_scale : boolean
        A flag which controls whether log-transformation is used for output.
        The default is `False`.
    cuda_used : boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.

    """
    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
    else:
        trained_record = None

    model = BuildingNetMTL_aux(input_channels=input_channels, input_size=input_size, aux_input_size=aux_input_size, backbone="ResNet", in_plane=in_plane,
                                    num_aux=num_aux, num_block=num_block, crossed=crossed, log_scale=log_scale,
                                    cuda_used=cuda_used, trained_record=trained_record)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter of ResNetMTL: ", total_num, " Trainable parameter of ResNetMTL: ", trainable_num)
    return model


# ************************* CNN instances using SENet as backbones *************************
def model_SEResNet(input_channels: int, input_size: int, in_plane: int, num_block: int, log_scale=False, activation="relu", cuda_used=False, **kwargs) -> BuildingNet:
    """Prepare SENet-STL for 3D building information extraction.

    Parameters
    ----------

    input_channels : int
        Number of input channels.
    input_size : int
        Size of input patches.
    in_plane : int
        Number of output channels after the initial convolutional layer.
    num_block : int
        Number of SEBlocks in each SELayer.
    log_scale : boolean
        A flag which controls whether log-transformation is used for output.
        The default is `False`.
    activation : str
        Activation function for model output.
        It can be chosen from: `relu`, `sigmoid`. The default is `relu`.
    cuda_used : boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.

    """
    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
    else:
        trained_record = None

    model = BuildingNet(input_channels=input_channels, input_size=input_size, backbone="SENet", in_plane=in_plane,
                        num_block=num_block, log_scale=log_scale, activation=activation, cuda_used=cuda_used, trained_record=trained_record)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter of SEResNet: ", total_num, " Trainable parameter of SEResNet: ", trainable_num)
    return model


def model_SEResNet_aux(input_channels: int, input_size: int, aux_input_size: int, in_plane: int, num_block: int, num_aux=1, log_scale=False, activation="relu", cuda_used=False, **kwargs) -> BuildingNet_aux:
    """Prepare SENet-STL using auxiliary input information for 3D building information extraction.

    Parameters
    ----------

    input_channels : int
        Number of input channels.
    input_size : int
        Size of input patches.
    aux_input_size : int
        Size of auxiliary input patches.
    in_plane : int
        Number of output channels after the initial convolutional layer.
    num_block : int
        Number of SEBlocks in each SELayer.
    num_aux : int
        Number of auxiliary variables.
        The default is `1`.
    log_scale : boolean
        A flag which controls whether log-transformation is used for output.
        The default is `False`.
    activation : str
        Activation function for model output.
        It can be chosen from: `relu`, `sigmoid`. The default is `relu`.
    cuda_used : boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.

    """
    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
    else:
        trained_record = None

    model = BuildingNet_aux(input_channels=input_channels, input_size=input_size, aux_input_size=aux_input_size, backbone="SENet", in_plane=in_plane,
                                num_aux=num_aux, num_block=num_block, log_scale=log_scale, activation=activation, 
                                cuda_used=cuda_used, trained_record=trained_record)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter of SEResNet: ", total_num, " Trainable parameter of SEResNet: ", trainable_num)
    return model


def model_SEResNetMTL(input_channels: int, input_size: int, in_plane: int, num_block: int, crossed=False, log_scale=False, cuda_used=False, **kwargs) -> BuildingNetMTL:
    """Prepare SENet-MTL using auxiliary input information for 3D building information extraction.

    Parameters
    ----------

    input_channels : int
        Number of input channels.
    input_size : int
        Size of input patches.
    in_plane : int
        Number of output channels after the initial convolutional layer.
    num_block : int
        Number of SEBlocks in each SELayer.
    crossed : boolean
        A flag which controls whether a link is to be added between last fully-connected layers of building footprint and height prediction.
        The default is `False`.
    log_scale : boolean
        A flag which controls whether log-transformation is used for output.
        The default is `False`.
    cuda_used : boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.

    """
    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
    else:
        trained_record = None

    model = BuildingNetMTL(input_channels=input_channels, input_size=input_size, backbone="SENet", in_plane=in_plane,
                           num_block=num_block, crossed=crossed, log_scale=log_scale,
                           cuda_used=cuda_used, trained_record=trained_record)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter of SEResNetMTL: ", total_num, " Trainable parameter of SEResNetMTL: ", trainable_num)
    return model


def model_SEResNetMTL_aux(input_channels: int, input_size: int, aux_input_size: int, in_plane: int, num_block: int, num_aux=1, crossed=False, log_scale=False, cuda_used=False, **kwargs) -> BuildingNetMTL_aux:
    """Prepare SENet-MTL using auxiliary input information for 3D building information extraction.

    Parameters
    ----------

    input_channels : int
        Number of input channels.
    input_size : int
        Size of input patches.
    aux_input_size : int
        Size of auxiliary input patches.
    in_plane : int
        Number of output channels after the initial convolutional layer.
    num_block : int
        Number of SEBlocks in each SELayer.
    num_aux : int
        Number of auxiliary variables.
        The default is `1`.
    crossed : boolean
        A flag which controls whether a link is to be added between last fully-connected layers of building footprint and height prediction.
        The default is `False`.
    log_scale : boolean
        A flag which controls whether log-transformation is used for output.
        The default is `False`.
    cuda_used : boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.

    """
    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
    else:
        trained_record = None

    model = BuildingNetMTL_aux(input_channels=input_channels, input_size=input_size, aux_input_size=aux_input_size, backbone="SENet", in_plane=in_plane,
                                    num_aux=num_aux, num_block=num_block, crossed=crossed, log_scale=log_scale,
                                    cuda_used=cuda_used, trained_record=trained_record)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter of SEResNetMTL: ", total_num, " Trainable parameter of SEResNetMTL: ", trainable_num)
    return model


# ************************* CNN instances using CBAM as backbones *************************
def model_CBAMResNet(input_channels: int, input_size: int, in_plane: int, num_block: int, log_scale=False, activation="relu", cuda_used=False, **kwargs) -> BuildingNet:
    """Prepare CBAM-STL for 3D building information extraction.

    Parameters
    ----------

    input_channels : int
        Number of input channels.
    input_size : int
        Size of input patches.
    in_plane : int
        Number of output channels after the initial convolutional layer.
    num_block : int
        Number of CBAMBlocks in each CBAMLayer.
    log_scale : boolean
        A flag which controls whether log-transformation is used for output.
        The default is `False`.
    activation : str
        Activation function for model output.
        It can be chosen from: `relu`, `sigmoid`. The default is `relu`.
    cuda_used : boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.

    """
    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
    else:
        trained_record = None

    model = BuildingNet(input_channels=input_channels, input_size=input_size, backbone="CBAM", in_plane=in_plane,
                        num_block=num_block, log_scale=log_scale, activation=activation, cuda_used=cuda_used, trained_record=trained_record)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter of CBAMResNet: ", total_num, " Trainable parameter of CBAMResNet: ", trainable_num)
    return model


def model_CBAMResNet_aux(input_channels: int, input_size: int, aux_input_size: int, in_plane: int, num_block: int, num_aux=1, log_scale=False, activation="relu", cuda_used=False, **kwargs) -> BuildingNet_aux:
    """Prepare CBAM-STL using auxiliary input information for 3D building information extraction.

    Parameters
    ----------

    input_channels : int
        Number of input channels.
    input_size : int
        Size of input patches.
    aux_input_size : int
        Size of auxiliary input patches.
    in_plane : int
        Number of output channels after the initial convolutional layer.
    num_block : int
        Number of CBAMBlocks in each CBAMLayer.
    num_aux : int
        Number of auxiliary variables.
        The default is `1`.
    log_scale : boolean
        A flag which controls whether log-transformation is used for output.
        The default is `False`.
    activation : str
        Activation function for model output.
        It can be chosen from: `relu`, `sigmoid`. The default is `relu`.
    cuda_used : boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.

    """
    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
    else:
        trained_record = None

    model = BuildingNet_aux(input_channels=input_channels, input_size=input_size, aux_input_size=aux_input_size, backbone="CBAM", in_plane=in_plane,
                                num_aux=num_aux, num_block=num_block, log_scale=log_scale, activation=activation, 
                                cuda_used=cuda_used, trained_record=trained_record)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter of CBAMResNet: ", total_num, " Trainable parameter of CBAMResNet: ", trainable_num)
    return model


def model_CBAMResNetMTL(input_channels: int, input_size: int, in_plane: int, num_block: int, crossed=False, log_scale=False, cuda_used=True, **kwargs) -> BuildingNetMTL:
    """Prepare CBAM-MTL using auxiliary input information for 3D building information extraction.

    Parameters
    ----------

    input_channels : int
        Number of input channels.
    input_size : int
        Size of input patches.
    in_plane : int
        Number of output channels after the initial convolutional layer.
    num_block : int
        Number of CBAMBlocks in each CBAMLayer.
    crossed : boolean
        A flag which controls whether a link is to be added between last fully-connected layers of building footprint and height prediction.
        The default is `False`.
    log_scale : boolean
        A flag which controls whether log-transformation is used for output.
        The default is `False`.
    cuda_used : boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.

    """
    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
    else:
        trained_record = None

    model = BuildingNetMTL(input_channels=input_channels, input_size=input_size, backbone="CBAM", in_plane=in_plane,
                           num_block=num_block, crossed=crossed, log_scale=log_scale,
                           cuda_used=cuda_used, trained_record=trained_record)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter of CBAMResNetMTL: ", total_num, " Trainable parameter of CBAMResNetMTL: ", trainable_num)
    return model


def model_CBAMResNetMTL_aux(input_channels: int, input_size: int, aux_input_size: int, in_plane: int, num_block: int, num_aux=1, crossed=False, log_scale=False, cuda_used=True, **kwargs) -> BuildingNetMTL_aux:
    """Prepare CBAM-MTL using auxiliary input information for 3D building information extraction.

    Parameters
    ----------

    input_channels : int
        Number of input channels.
    input_size : int
        Size of input patches.
    aux_input_size : int
        Size of auxiliary input patches.
    in_plane : int
        Number of output channels after the initial convolutional layer.
    num_block : int
        Number of CBAMBlocks in each CBAMLayer.
    num_aux : int
        Number of auxiliary variables.
        The default is `1`.
    crossed : boolean
        A flag which controls whether a link is to be added between last fully-connected layers of building footprint and height prediction.
        The default is `False`.
    log_scale : boolean
        A flag which controls whether log-transformation is used for output.
        The default is `False`.
    cuda_used : boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.

    """
    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
    else:
        trained_record = None

    model = BuildingNetMTL_aux(input_channels=input_channels, input_size=input_size, aux_input_size=aux_input_size, backbone="CBAM", in_plane=in_plane,
                                    num_aux=num_aux, num_block=num_block, crossed=crossed, log_scale=log_scale,
                                    cuda_used=cuda_used, trained_record=trained_record)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter of CBAMResNetMTL: ", total_num, " Trainable parameter of CBAMResNetMTL: ", trainable_num)
    return model


if __name__ == "__main__":
    #pretrained_weight = os.path.join("DL_run", "res_file", "check_pt_cbam_100m", "experiment_2", "checkpoint.pth.tar")

    # m = model_ResNet(in_plane=64, input_channels=6, input_size=30, num_block=4)
    # m = model_ResNet_aux(in_plane=64, input_channels=6, input_size=30, aux_input_size=30, num_block=1, num_aux=1)
    # m = model_ResNetMTL(in_plane=64, input_channels=6, input_size=30, num_block=4)
    m = model_ResNetMTL_aux(in_plane=64, input_channels=6, input_size=30, aux_input_size=30, num_block=1, num_aux=1)
    # m = model_SEResNet(in_plane=64, input_channels=6, input_size=15, num_block=4, trained_record=pretrained_weight)
    # m = model_CBAMResNet(in_plane=64, input_channels=6, input_size=60, num_block=3, trained_record=pretrained_weight)

    m.eval()
    '''
    for name, param in m.state_dict().items():
        print(name)
    '''

    test_dta = torch.ones(8, 6, 120, 120)

    test_aux = torch.ones(8, 1, 120, 120) * 2
    test_aux[1] = test_aux[1] * 2
    test_aux[2] = test_aux[2] * 4
    test_aux[4] = test_aux[4] * 8

    # test_out = m(test_dta)
    # test_out = m(test_dta, test_aux)
    # test_out, test_out_b = m(test_dta)
    test_out, test_out_b = m(test_dta, test_aux)
    print(test_out)
    print(test_out_b)
