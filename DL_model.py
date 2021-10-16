import os
import math
from DL_module import *


# ************************* Backbones *************************
class ResNetBackbone(nn.Module):
    def __init__(self, input_channels, input_size, in_plane=64, num_block=2):
        super(ResNetBackbone, self).__init__()

        self.in_plane = in_plane
        self.bn1 = nn.BatchNorm2d(self.in_plane)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = None
        if input_size in [120]:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=7, stride=2, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        elif input_size in [60]:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=5, stride=2, padding=1, bias=False)
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

    def _make_layer(self, num_plane, blocks, stride=1):
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
    def __init__(self, input_channels, input_size, in_plane=64, num_block=2):
        super(SENetBackbone, self).__init__()
        self.in_plane = in_plane
        self.bn1 = nn.BatchNorm2d(self.in_plane)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = None
        if input_size in [120]:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=5, stride=2, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        elif input_size in [60]:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=5, stride=2, padding=1, bias=False)
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

    def _make_layer(self, num_plane, blocks, stride=1):
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
    def __init__(self, input_channels, input_size, in_plane=64, num_block=2):
        super(CBAMBackbone, self).__init__()
        self.in_plane = in_plane
        self.bn1 = nn.BatchNorm2d(self.in_plane)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = None
        if input_size in [120]:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        elif input_size in [60]:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.in_plane, kernel_size=3, stride=1, padding=1, bias=False)
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

    def _make_layer(self, num_plane, blocks, stride=1):
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
    def __init__(self, input_channels, input_size, backbone, in_plane=64, num_block=3, log_scale=False, activation="relu", cuda_used=True, **kwargs):
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
        x = self.fc_out(x)

        # ---For BuildingFootprint prediction, we set log_scale=False by default
        if not self.log_scale:
            x = self.activation(x)

        return x

    def load_pretrained_model(self, trained_record):
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


# ************************* CNNs in their Multi-Task Learning version *************************
class BuildingNetMTL(nn.Module):
    def __init__(self, input_channels, input_size, backbone, in_plane=64, num_block=3, crossed=False, log_scale=False, cuda_used=True, **kwargs):
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

    def load_pretrained_model(self, trained_record):
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


def model_ResNet(input_channels, input_size, in_plane, num_block, log_scale=False, activation="relu", cuda_used=True, **kwargs):
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


def model_ResNetMTL(input_channels, input_size, in_plane, num_block, crossed=False, log_scale=False, cuda_used=True, **kwargs):
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


def model_SEResNet(input_channels, input_size, in_plane, num_block, log_scale=False, activation="relu", cuda_used=True, **kwargs):
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


def model_SEResNetMTL(input_channels, input_size, in_plane, num_block, crossed=False, log_scale=False, cuda_used=True, **kwargs):
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


def model_CBAMResNet(input_channels, input_size, in_plane, num_block, log_scale=False, activation="relu", cuda_used=True, **kwargs):
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


def model_CBAMResNetMTL(input_channels, input_size, in_plane, num_block, crossed=False, log_scale=False, cuda_used=True, **kwargs):
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


if __name__ == "__main__":
    pretrained_weight = os.path.join("DL_run", "res_file", "check_pt_cbam_100m", "experiment_2", "checkpoint.pth.tar")

    # m = model_ResNet(in_plane=64, input_channels=6, input_size=30, num_block=4)
    # m = model_ResNetMTL(in_plane=64, input_channels=6, input_size=30, num_block=4)
    # m = model_SEResNet(in_plane=64, input_channels=6, input_size=15, num_block=4, trained_record=pretrained_weight)
    m = model_CBAMResNet(in_plane=64, input_channels=6, input_size=60, num_block=3, trained_record=pretrained_weight)

    m.eval()
    '''
    for name, param in m.state_dict().items():
        print(name)
    '''

    test_dta = torch.ones(8, 6, 30, 30)
    test_out = m(test_dta)
    # test_out, test_out_b = m(test_dta)
    print(test_out)
