import torch
import torch.nn as nn


def downSamplingChoice(in_plane, out_plane, stride):
    if (stride != 1) or (in_plane != out_plane):
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_plane, out_channels=out_plane, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_plane)
        )
    else:
        downsample = None
    return downsample


# ************************* CNN's Module *************************
# ---basic ResBlock used in ResNet
# ---ref to: He, K., Zhang, X., Ren, S., & Sun, J. (2016).
# ------Deep Residual Learning for Image Recognition.
# ------In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 770–778).
class BasicResBlock(nn.Module):

    def __init__(self, in_plane, num_plane, stride=1, downsample=None):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_plane, out_channels=num_plane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_plane, out_channels=num_plane, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_plane)
        self.downsample = downsample

    def forward(self, x):
        # [H, W] -> [H/s, W/s]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # [H/s, W/s] -> [H/s, W/s]
        out = self.conv2(out)
        out = self.bn2(out)

        # The residual and the output must be of the same dimensions
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out


# ---ResBlock with channel attention used in SENet
# ---ref to: Hu, J., Shen, L., & Sun, G. (2018).
# ------Squeeze-and-Excitation Networks.
# ------In 2018 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 7132–7141).
class SEResBlock(nn.Module):

    def __init__(self, in_plane, num_plane, stride=1, reduction=16, downsample=None):
        super(SEResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_plane, out_channels=num_plane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_plane, out_channels=num_plane, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_plane)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.SEfc = nn.Sequential(
            nn.Linear(num_plane, num_plane // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_plane // reduction, num_plane, bias=False),
            nn.Sigmoid()
        )
        self.downsample = downsample

    def forward(self, x):
        # [H, W] -> [H/s, W/s]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # [H/s, W/s] -> [H/s, W/s]
        out = self.conv2(out)
        out = self.bn2(out)
        channel_att = self.avg_pool(out)
        channel_att = torch.flatten(channel_att, start_dim=1)
        channel_att = self.SEfc(channel_att)
        channel_att = channel_att.view([channel_att.size(0), channel_att.size(1), 1, 1])
        out = out * channel_att.expand_as(out)
        # The residual and the output must be of the same dimensions
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out


# ---CBAM (Convolutional Block Attention Module)
# ---ref to: Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018).
# ------CBAM: Convolutional Block Attention Module.
# ------In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 3–19).
class ChannelAttentionModule(nn.Module):
    def __init__(self, num_plane, reduction=16, mapping="conv"):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.mapping = mapping
        if self.mapping == "mlp":
            self.shared_mapping = nn.Sequential(
                nn.Linear(num_plane, num_plane // reduction),
                nn.ReLU(),
                nn.Linear(num_plane // reduction, num_plane)
            )
        elif self.mapping == "conv":
            self.shared_mapping = nn.Sequential(
                nn.Conv2d(in_channels=num_plane, out_channels=num_plane // reduction, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_plane // reduction, out_channels=num_plane, kernel_size=1, bias=False)
            )
        else:
            raise NotImplementedError("Unknown shared mapping in CBAM")

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_feat = self.avg_pool(x)
        max_pool_feat = self.max_pool(x)

        # ------convert (N, C, H=1, W=1) to (N, C) for MLP layer
        if self.mapping == "mlp":
            avg_pool_feat = torch.flatten(avg_pool_feat, start_dim=1)
            max_pool_feat = torch.flatten(max_pool_feat, start_dim=1)

        channel_att = self.shared_mapping(avg_pool_feat) + self.shared_mapping(max_pool_feat)
        channel_att = self.sigmoid(channel_att)
        channel_att = channel_att.view([channel_att.size(0), channel_att.size(1), 1, 1])

        out = x * channel_att.expand_as(x)
        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(num_features=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_feat = torch.max(x, 1)[0]
        mean_feat = torch.mean(x, 1)
        feat = torch.cat((max_feat.unsqueeze(1), mean_feat.unsqueeze(1)), dim=1)
        feat = self.conv(feat)
        spatial_att = self.sigmoid(feat)

        out = x * spatial_att
        return out


class CBAMResBlock(nn.Module):

    def __init__(self, in_plane, num_plane, stride=1, reduction=16, downsample=None):
        super(CBAMResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_plane, out_channels=num_plane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_plane, out_channels=num_plane, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_plane)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.cbam = nn.Sequential(
            ChannelAttentionModule(num_plane, reduction),
            SpatialAttentionModule(kernel_size=5)
        )
        self.downsample = downsample

    def forward(self, x):
        # [H, W] -> [H/s, W/s]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # [H/s, W/s] -> [H/s, W/s]
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        # The residual and the output must be of the same dimensions
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out


# ************************* Loss Function Module *************************
class RegressionLosses(object):
    def __init__(self, cuda=False):
        self.cuda = cuda

    def build_loss(self, loss_mode="MSE"):
        if loss_mode == "MSE":
            return self.MSELoss
        elif loss_mode == "Huber" or loss_mode == "AdaptiveHuber":
            return self.HuberLoss
        else:
            raise NotImplementedError

    def MSELoss(self, val_pred, val_true, beta):
        criterion = nn.MSELoss(reduction="mean")
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(val_pred, val_true.float())

        return loss

    def HuberLoss(self, val_pred, val_true, beta):
        criterion = nn.SmoothL1Loss(reduction="mean", beta=beta)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(val_pred, val_true.float())

        return loss


class RegressionLosses_MTL(object):
    def __init__(self, cuda=False, adaptive_weight=True):
        self.cuda = cuda
        self.adaptive_weight = adaptive_weight

    def build_loss(self, loss_mode="MSE"):
        if loss_mode == "MSE":
            return self.MSELoss_MTL
        elif loss_mode == "Huber" or loss_mode == "AdaptiveHuber":
            return self.HuberLoss_MTL
        else:
            raise NotImplementedError

    def MSELoss_MTL(self, footprint_pred, footprint_true, height_pred, height_true, beta_footprint, beta_height, s_footprint, s_height):
        criterion = nn.MSELoss(reduction="mean")
        if self.cuda:
            criterion = criterion.cuda()

        loss_footprint = criterion(footprint_pred, footprint_true.float())
        loss_height = criterion(height_pred, height_true.float())

        if self.adaptive_weight:
            w_footprint = 0.5 * torch.exp(-s_footprint)
            r_footprint = 0.5 * s_footprint
            w_height = 0.5 * torch.exp(-s_height)
            r_height = 0.5 * s_height
            loss = w_footprint * loss_footprint + r_footprint + w_height * loss_height + r_height
        else:
            loss = s_footprint * loss_footprint + s_height * loss_height

        return loss

    def HuberLoss_MTL(self, footprint_pred, footprint_true,  height_pred, height_true, beta_footprint, beta_height, s_footprint, s_height):
        criterion_footprint = nn.SmoothL1Loss(reduction="mean", beta=beta_footprint)
        criterion_height = nn.SmoothL1Loss(reduction="mean", beta=beta_height)
        if self.cuda:
            criterion_footprint = criterion_footprint.cuda()
            criterion_height = criterion_height.cuda()

        loss_footprint = criterion_footprint(footprint_pred, footprint_true.float())
        loss_height = criterion_height(height_pred, height_true.float())

        if self.adaptive_weight:
            w_footprint = 0.5 * torch.exp(-s_footprint)
            r_footprint = 0.5 * s_footprint
            w_height = 0.5 * torch.exp(-s_height)
            r_height = 0.5 * s_height
            loss = w_footprint * loss_footprint + r_footprint + w_height * loss_height + r_height
        else:
            loss = s_footprint * loss_footprint + s_height * loss_height

        return loss
