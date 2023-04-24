import os
import math
import numpy as np
import torch
import tensorflow as tf
from keras import layers
from keras.utils.layer_utils import count_params


kernel_size_ref = {100: 20, 250: 40, 500: 80, 1000: 160}
overlapSize_ref = {100: 10, 250: 15, 500: 30, 1000: 60}
degree_ref = {100: 0.0009, 250: 0.00225, 500: 0.0045, 1000: 0.009}

# ************************* Utility *************************
def get_state_dict_structure(torch_state_dict):
    para_structure = {}
    name_list = list(torch_state_dict.keys())
    name_depth = np.array([len(n.split(".")) for n in name_list])
    depth_max = max(name_depth)

    for d in range(1, depth_max):
        name_level_d_id = np.where(name_depth == int(d+1))[0]
        # ------we only need the name of layer/module rather than a variable/parameter,
        # ---------e.g., for 'features.res_layer.2.0.bn1.weight', 'features.res_layer.2.0.bn1.bias', we only keep
        # ---------'features.res_layer.2.0.bn1'
        para_structure["L"+str(d)] = np.unique([".".join(name_list[idx].split(".")[:-1]) for idx in name_level_d_id])

    return para_structure


def get_name_mapping_torch2tf(torch_state_dict):
    state_torch = get_state_dict_structure(torch_state_dict)


# ************************* CNN's Module with Tensorflow 2.0 implementation *************************
class BasicResBlock_tf(layers.Layer):

    def __init__(self, in_plane, num_plane, stride=1, downsample=None, **kwargs):
        super(BasicResBlock_tf, self).__init__(**kwargs)

        if "name" in kwargs.keys():
            prev_name = kwargs["name"]
        else:
            prev_name = None

        self.pad1 = layers.ZeroPadding2D(padding=(1, 1))
        self.conv1 = layers.Conv2D(filters=num_plane, kernel_size=3, strides=stride, padding="valid", use_bias=False,
                                   name=".".join([prev_name, "conv1"]))
        self.bn1 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5,
                                             name=".".join([prev_name, "bn1"]))
        self.relu = layers.ReLU()
        self.pad2 = layers.ZeroPadding2D(padding=(1, 1))
        self.conv2 = layers.Conv2D(filters=num_plane, kernel_size=3, padding="valid", use_bias=False,
                                   name=".".join([prev_name, "conv2"]))
        self.bn2 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5,
                                             name=".".join([prev_name, "bn2"]))
        self.downsample = downsample

    def call(self, inputs, training=None):
        out = self.pad1(inputs)
        out = self.conv1(out)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(inputs)
        else:
            identity = inputs

        out = layers.add([out, identity])
        out = self.relu(out)

        return out


class SEResBlock_tf(layers.Layer):

    def __init__(self, in_plane, num_plane, stride=1, reduction=16, downsample=None, **kwargs):
        super(SEResBlock_tf, self).__init__(**kwargs)

        if "name" in kwargs.keys():
            prev_name = kwargs["name"]
        else:
            prev_name = None

        self.pad1 = layers.ZeroPadding2D(padding=(1, 1))
        self.conv1 = layers.Conv2D(filters=num_plane, kernel_size=3, strides=stride, padding="valid", use_bias=False,
                                   name=".".join([prev_name, "conv1"]))
        self.bn1 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5,
                                             name=".".join([prev_name, "bn1"]))
        self.relu = layers.ReLU()
        self.pad2 = layers.ZeroPadding2D(padding=(1, 1))
        self.conv2 = layers.Conv2D(filters=num_plane, kernel_size=3, padding="valid", use_bias=False,
                                   name=".".join([prev_name, "conv2"]))
        self.bn2 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5,
                                             name=".".join([prev_name, "bn2"]))
        self.avgpool = layers.GlobalAveragePooling2D()

        self.SEfc = tf.keras.Sequential()
        self.SEfc.add(layers.Dense(units=num_plane // reduction, use_bias=False,
                                   name=".".join([prev_name, "SEfc", "0"])))
        self.SEfc.add(layers.ReLU())
        self.SEfc.add(layers.Dense(units=num_plane, use_bias=False,
                                   name=".".join([prev_name, "SEfc", "2"])))

        self.downsample = downsample

    def call(self, inputs, training=None):
        out = self.pad1(inputs)
        out = self.conv1(out)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)

        channel_att = self.avgpool(out)
        channel_att = self.SEfc(channel_att)
        channel_att = tf.keras.activations.sigmoid(channel_att)
        channel_att = tf.reshape(channel_att, shape=[-1, 1, 1, channel_att.shape[-1]])
        out = out * channel_att

        if self.downsample is not None:
            identity = self.downsample(inputs)
        else:
            identity = inputs

        out = layers.add([out, identity])
        out = self.relu(out)

        return out


# ************************* Backbones with Tensorflow 2.0 implementation *************************
class ResNetBackbone_tf(layers.Layer):
    def __init__(self, input_channels, input_size, in_plane=64, num_block=2, **kwargs):
        super(ResNetBackbone_tf, self).__init__(**kwargs)

        if "name" in kwargs.keys():
            prev_name = kwargs["name"]
        else:
            prev_name = None

        self.in_plane = in_plane
        self.conv1 = layers.Conv2D(filters=self.in_plane, kernel_size=3, strides=1, padding="same", use_bias=False,
                                   name=".".join([prev_name, "conv1"]))
        self.bn1 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=".".join([prev_name, "bn1"]))
        self.relu = layers.ReLU()

        self.maxpool = None
        if input_size in [120]:
            self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding="same")
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 2)
        else:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 1)

        layer_id = 0
        self.res_layer = tf.keras.Sequential()
        self.res_layer.add(self._make_layer(in_plane, blocks=num_block,
                                            name_prefix=".".join([prev_name, "res_layer", str(layer_id)])))

        if num_reduce == 1:
            in_plane = in_plane * 2
            layer_id += 1
            self.res_layer.add(self._make_layer(in_plane, blocks=num_block,
                                                name_prefix=".".join([prev_name, "res_layer", str(layer_id)])))

        for i in range(0, num_reduce):
            num_plane = int(in_plane * math.pow(2, i+1))
            layer_id += 1
            self.res_layer.add(self._make_layer(num_plane, blocks=num_block, stride=2,
                                                name_prefix=".".join([prev_name, "res_layer", str(layer_id)])))

    def _make_layer(self, num_plane, blocks, stride=1, name_prefix=None):
        block_id = 0
        downsample = None
        if (stride != 1) or (self.in_plane != num_plane):
            downsample = tf.keras.Sequential()
            downsample.add(layers.Conv2D(filters=num_plane, kernel_size=1, strides=stride, use_bias=False,
                                         name=".".join([name_prefix, str(block_id), "downsample", "0"])))
            downsample.add(layers.BatchNormalization(momentum=0.1, epsilon=1e-5,
                                                     name=".".join([name_prefix, str(block_id), "downsample", "1"])))

        tmp_layers = tf.keras.Sequential()
        tmp_layers.add(BasicResBlock_tf(self.in_plane, num_plane, stride, downsample=downsample,
                                        name=".".join([name_prefix, str(block_id)])))
        self.in_plane = num_plane
        for block_id in range(1, blocks):
            tmp_layers.add(BasicResBlock_tf(self.in_plane, num_plane, name=".".join([name_prefix, str(block_id)])))

        return tmp_layers

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        if self.maxpool is not None:
            out = self.maxpool(out)

        out = self.res_layer(out)

        return out


class SENetBackbone_tf(layers.Layer):
    def __init__(self, input_channels, input_size, in_plane=64, num_block=2, **kwargs):
        super(SENetBackbone_tf, self).__init__(**kwargs)

        if "name" in kwargs.keys():
            prev_name = kwargs["name"]
        else:
            prev_name = None

        self.in_plane = in_plane
        
        self.bn1 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=".".join([prev_name, "bn1"]))
        self.relu = layers.ReLU()

        self.maxpool = None
        if input_size in [120, 160]:
            self.conv1 = layers.Conv2D(filters=self.in_plane, kernel_size=7, strides=2, padding="same", use_bias=False,
                                            name=".".join([prev_name, "conv1"]))
            self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding="same")
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        elif input_size in [60, 80]:
            self.conv1 = layers.Conv2D(filters=self.in_plane, kernel_size=7, strides=2, padding="same", use_bias=False,
                                            name=".".join([prev_name, "conv1"]))
            self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding="same")
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        elif input_size in [40]:
            self.conv1 = layers.Conv2D(filters=self.in_plane, kernel_size=7, strides=2, padding="same", use_bias=False,
                                            name=".".join([prev_name, "conv1"]))
            self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding="same")
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
        else:
            self.conv1 = layers.Conv2D(filters=self.in_plane, kernel_size=3, strides=1, padding="same", use_bias=False,
                                            name=".".join([prev_name, "conv1"]))
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 1)

        layer_id = 0
        self.res_layer = tf.keras.Sequential()
        self.res_layer.add(self._make_layer(in_plane, blocks=num_block,
                                            name_prefix=".".join([prev_name, "res_layer", str(layer_id)])))

        if num_reduce == 1:
            in_plane = in_plane * 2
            layer_id += 1
            self.res_layer.add(self._make_layer(in_plane, blocks=num_block,
                                                name_prefix=".".join([prev_name, "res_layer", str(layer_id)])))

        for i in range(0, num_reduce):
            num_plane = int(in_plane * math.pow(2, i+1))
            layer_id += 1
            self.res_layer.add(self._make_layer(num_plane, blocks=num_block, stride=2,
                                                name_prefix=".".join([prev_name, "res_layer", str(layer_id)])))

    def _make_layer(self, num_plane, blocks, stride=1, name_prefix=None):
        block_id = 0
        downsample = None
        if (stride != 1) or (self.in_plane != num_plane):
            downsample = tf.keras.Sequential()
            downsample.add(layers.Conv2D(filters=num_plane, kernel_size=1, strides=stride, use_bias=False,
                                         padding="valid",
                                         name=".".join([name_prefix, str(block_id), "downsample", "0"])))
            downsample.add(layers.BatchNormalization(momentum=0.1, epsilon=1e-5,
                                                     name=".".join([name_prefix, str(block_id), "downsample", "1"])))

        tmp_layers = tf.keras.Sequential()
        tmp_layers.add(SEResBlock_tf(self.in_plane, num_plane, stride, downsample=downsample,
                                     name=".".join([name_prefix, str(block_id)])))
        self.in_plane = num_plane
        for block_id in range(1, blocks):
            tmp_layers.add(SEResBlock_tf(self.in_plane, num_plane,
                                         name=".".join([name_prefix, str(block_id)])))

        return tmp_layers

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        if self.maxpool is not None:
            out = self.maxpool(out)

        out = self.res_layer(out)

        return out


# ************************* CNNs with Tensorflow 2.0 implementation *************************
class BuildingNet_aux_tf(tf.keras.Model):

    def __init__(self, input_channels: int, input_size: int, aux_input_size: int, in_plane=64, num_aux=1, num_block=2, log_scale=False, activation="relu", cuda_used=True, **kwargs):
        super(BuildingNet_aux_tf, self).__init__()
        self.cuda_used = cuda_used
        aux_in_plane = int(in_plane / 4.0)
        
        self.features = SENetBackbone_tf(input_channels, input_size, in_plane=in_plane, num_block=num_block, name="features")
        self.aux_features = SENetBackbone_tf(num_aux, aux_input_size, in_plane=aux_in_plane, num_block=num_block, name="aux_features")
        
        self.log_scale = log_scale
        self.avgpool = layers.GlobalAveragePooling2D()
        self.relu = layers.ReLU()

        if input_size in [120, 160, 80]:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
            aux_num_reduce = int(math.floor(math.log(aux_input_size / 2, 2)) - 3)
        elif input_size in [60, 40]:
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

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(units=int(num_plane / 2), name="fc")
        self.bn_out = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name="bn_out")
        self.fc_out = layers.Dense(units=1, name="fc_out")

        if activation == "relu":
            self.activation = layers.ReLU()
        elif activation == "sigmoid":
            self.activation = layers.Activation(activation)
        else:
            raise NotImplementedError

    def call(self, dta_input, training=None):
        x  = dta_input[:, :, :, :-1]
        aux = tf.expand_dims(dta_input[:, :, :, -1], axis=-1)

        x = self.features(x, training=training)
        x = self.avgpool(x)
        x = self.flatten(x)

        aux = self.aux_features(aux, training=training)
        aux = self.avgpool(aux)
        aux = self.flatten(aux)

        x = tf.concat([x, aux], axis=1)

        x = self.fc(x)
        x = self.bn_out(x, training=training)
        x = self.relu(x)
        x = self.fc_out(x)

        if not self.log_scale:
            x = self.activation(x)

        return x

    def get_para_shape(self):
        for l in self.layers:
            print([tensor.shape for tensor in l.get_weights()])

    def load_pretrained_model(self, trained_record):
        print("Loading the PyTorch model from: " + trained_record)
        if self.cuda_used:
            torch_state_dict = torch.load(trained_record)["state_dict"]
            # ------initialize fc layer
            tf_fc = self.get_layer("fc")
            weights = torch_state_dict["fc.weight"].cpu().numpy()
            bias = torch_state_dict["fc.bias"].cpu().numpy()
            tf_fc.set_weights([weights.transpose((1, 0)), bias])
            # ------initialize bn_out layer
            tf_bn_out = self.get_layer("bn_out")
            gamma = torch_state_dict["bn_out.weight"].cpu().numpy()
            beta = torch_state_dict["bn_out.bias"].cpu().numpy()
            mean = torch_state_dict["bn_out.running_mean"].cpu().numpy()
            var = torch_state_dict["bn_out.running_var"].cpu().numpy()
            tf_bn_out.set_weights([gamma, beta, mean, var])
            # ------initialize fc_out layer
            tf_fc_out = self.get_layer("fc_out")
            weights = torch_state_dict["fc_out.weight"].cpu().numpy()
            bias = torch_state_dict["fc_out.bias"].cpu().numpy()
            tf_fc_out.set_weights([weights.transpose((1, 0)), bias])
            # ------initialize backbone layer
            for bkbone in ["features", "aux_features"]:
                tf_backbone = self.get_layer(bkbone)
                # ---------[1] initialize the initial convolutional layer in the backbone layer
                tf_conv1 = tf_backbone.conv1
                weights = torch_state_dict[tf_conv1.name + ".weight"].cpu().numpy()
                weights = weights.transpose((2, 3, 1, 0))
                tf_conv1.set_weights([weights])
                # ---------[2] initialize the initial BatchNormalization layer in the backbone layer
                tf_bn1 = tf_backbone.bn1
                gamma = torch_state_dict[tf_bn1.name + ".weight"].cpu().numpy()
                beta = torch_state_dict[tf_bn1.name + ".bias"].cpu().numpy()
                mean = torch_state_dict[tf_bn1.name + ".running_mean"].cpu().numpy()
                var = torch_state_dict[tf_bn1.name + ".running_var"].cpu().numpy()
                tf_bn1.set_weights([gamma, beta, mean, var])
                # ---------[3] initialize the ResLayer in the backbone layer
                tf_res_layer = tf_backbone.res_layer
                # ---------------iteration over multiple res_layers
                for seq_layer in tf_res_layer.layers:
                    # ------------------iteration over multiple res_blocks in one res_layer
                    for res_block in seq_layer.layers:
                        # ---------------------iteration over components in one res_block
                        # ------------------------conv1
                        base_conv1 = res_block.conv1
                        weights = torch_state_dict[base_conv1.name + ".weight"].cpu().numpy()
                        weights = weights.transpose((2, 3, 1, 0))
                        base_conv1.set_weights([weights])
                        # ------------------------bn1
                        base_bn1 = res_block.bn1
                        gamma = torch_state_dict[base_bn1.name + ".weight"].cpu().numpy()
                        beta = torch_state_dict[base_bn1.name + ".bias"].cpu().numpy()
                        mean = torch_state_dict[base_bn1.name + ".running_mean"].cpu().numpy()
                        var = torch_state_dict[base_bn1.name + ".running_var"].cpu().numpy()
                        base_bn1.set_weights([gamma, beta, mean, var])
                        # ------------------------conv2
                        base_conv2 = res_block.conv2
                        weights = torch_state_dict[base_conv2.name + ".weight"].cpu().numpy()
                        weights = weights.transpose((2, 3, 1, 0))
                        base_conv2.set_weights([weights])
                        # ------------------------bn2
                        base_bn2 = res_block.bn2
                        gamma = torch_state_dict[base_bn2.name + ".weight"].cpu().numpy()
                        beta = torch_state_dict[base_bn2.name + ".bias"].cpu().numpy()
                        mean = torch_state_dict[base_bn2.name + ".running_mean"].cpu().numpy()
                        var = torch_state_dict[base_bn2.name + ".running_var"].cpu().numpy()
                        base_bn2.set_weights([gamma, beta, mean, var])
                        # ------------------------downsample
                        if res_block.downsample is not None:
                            base_downsample = res_block.downsample
                            for l in base_downsample.layers:
                                if "downsample.0" in l.name:
                                    base_conv = base_downsample.get_layer(l.name)
                                    weights = torch_state_dict[l.name + ".weight"].cpu().numpy()
                                    weights = weights.transpose((2, 3, 1, 0))
                                    base_conv.set_weights([weights])
                                elif "downsample.1" in l.name:
                                    base_bn = base_downsample.get_layer(l.name)
                                    gamma = torch_state_dict[l.name + ".weight"].cpu().numpy()
                                    beta = torch_state_dict[l.name + ".bias"].cpu().numpy()
                                    mean = torch_state_dict[l.name + ".running_mean"].cpu().numpy()
                                    var = torch_state_dict[l.name + ".running_var"].cpu().numpy()
                                    base_bn.set_weights([gamma, beta, mean, var])
                        # ------------------------SEfc
                        if hasattr(res_block, "SEfc"):
                            base_sefc = res_block.SEfc
                            for l in base_sefc.layers:
                                if "fc" in l.name:
                                    base_fc = base_sefc.get_layer(l.name)
                                    weights = torch_state_dict[l.name + ".weight"].cpu().numpy()
                                    base_fc.set_weights([weights.transpose((1, 0))])
        else:
            torch_state_dict = torch.load(trained_record, map_location=torch.device('cpu'))["state_dict"]
            # ------initialize fc layer
            tf_fc = self.get_layer("fc")
            weights = torch_state_dict["fc.weight"].numpy()
            bias = torch_state_dict["fc.bias"].numpy()
            tf_fc.set_weights([weights.transpose((1, 0)), bias])
            # ------initialize bn_out layer
            tf_bn_out = self.get_layer("bn_out")
            gamma = torch_state_dict["bn_out.weight"].numpy()
            beta = torch_state_dict["bn_out.bias"].numpy()
            mean = torch_state_dict["bn_out.running_mean"].numpy()
            var = torch_state_dict["bn_out.running_var"].numpy()
            tf_bn_out.set_weights([gamma, beta, mean, var])
            # ------initialize fc_out layer
            tf_fc_out = self.get_layer("fc_out")
            weights = torch_state_dict["fc_out.weight"].numpy()
            bias = torch_state_dict["fc_out.bias"].numpy()
            tf_fc_out.set_weights([weights.transpose((1, 0)), bias])
            # ------initialize backbone layer
            for bkbone in ["features", "aux_features"]:
                tf_backbone = self.get_layer(bkbone)
                # ---------[1] initialize the initial convolutional layer in the backbone layer
                tf_conv1 = tf_backbone.conv1
                weights = torch_state_dict[tf_conv1.name + ".weight"].numpy()
                weights = weights.transpose((2, 3, 1, 0))
                tf_conv1.set_weights([weights])
                # ---------[2] initialize the initial BatchNormalization layer in the backbone layer
                tf_bn1 = tf_backbone.bn1
                gamma = torch_state_dict[tf_bn1.name + ".weight"].numpy()
                beta = torch_state_dict[tf_bn1.name + ".bias"].numpy()
                mean = torch_state_dict[tf_bn1.name + ".running_mean"].numpy()
                var = torch_state_dict[tf_bn1.name + ".running_var"].numpy()
                tf_bn1.set_weights([gamma, beta, mean, var])
                # ---------[3] initialize the ResLayer in the backbone layer
                tf_res_layer = tf_backbone.res_layer
                # ---------------iteration over multiple res_layers
                for seq_layer in tf_res_layer.layers:
                    # ------------------iteration over multiple res_blocks in one res_layer
                    for res_block in seq_layer.layers:
                        # ---------------------iteration over components in one res_block
                        # ------------------------conv1
                        base_conv1 = res_block.conv1
                        weights = torch_state_dict[base_conv1.name + ".weight"].numpy()
                        weights = weights.transpose((2, 3, 1, 0))
                        base_conv1.set_weights([weights])
                        # ------------------------bn1
                        base_bn1 = res_block.bn1
                        gamma = torch_state_dict[base_bn1.name + ".weight"].numpy()
                        beta = torch_state_dict[base_bn1.name + ".bias"].numpy()
                        mean = torch_state_dict[base_bn1.name + ".running_mean"].numpy()
                        var = torch_state_dict[base_bn1.name + ".running_var"].numpy()
                        base_bn1.set_weights([gamma, beta, mean, var])
                        # ------------------------conv2
                        base_conv2 = res_block.conv2
                        weights = torch_state_dict[base_conv2.name + ".weight"].numpy()
                        weights = weights.transpose((2, 3, 1, 0))
                        base_conv2.set_weights([weights])
                        # ------------------------bn2
                        base_bn2 = res_block.bn2
                        gamma = torch_state_dict[base_bn2.name + ".weight"].numpy()
                        beta = torch_state_dict[base_bn2.name + ".bias"].numpy()
                        mean = torch_state_dict[base_bn2.name + ".running_mean"].numpy()
                        var = torch_state_dict[base_bn2.name + ".running_var"].numpy()
                        base_bn2.set_weights([gamma, beta, mean, var])
                        # ------------------------downsample
                        if res_block.downsample is not None:
                            base_downsample = res_block.downsample
                            for l in base_downsample.layers:
                                if "downsample.0" in l.name:
                                    base_conv = base_downsample.get_layer(l.name)
                                    weights = torch_state_dict[l.name + ".weight"].numpy()
                                    weights = weights.transpose((2, 3, 1, 0))
                                    base_conv.set_weights([weights])
                                elif "downsample.1" in l.name:
                                    base_bn = base_downsample.get_layer(l.name)
                                    gamma = torch_state_dict[l.name + ".weight"].numpy()
                                    beta = torch_state_dict[l.name + ".bias"].numpy()
                                    mean = torch_state_dict[l.name + ".running_mean"].numpy()
                                    var = torch_state_dict[l.name + ".running_var"].numpy()
                                    base_bn.set_weights([gamma, beta, mean, var])
                        # ------------------------SEfc
                        if hasattr(res_block, "SEfc"):
                            base_sefc = res_block.SEfc
                            for l in base_sefc.layers:
                                if "fc" in l.name:
                                    base_fc = base_sefc.get_layer(l.name)
                                    weights = torch_state_dict[l.name + ".weight"].numpy()
                                    base_fc.set_weights([weights.transpose((1, 0))])


class BuildingNetMTL_aux_tf(tf.keras.Model):
    def __init__(self, input_channels: int, input_size: int, aux_input_size: int, in_plane=64, num_aux=1, num_block=2, crossed=False, log_scale=False, cuda_used=True, **kwargs):
        super(BuildingNetMTL_aux_tf, self).__init__()
        self.crossed = crossed
        self.cuda_used = cuda_used
        aux_in_plane = int(in_plane / 4.0)
        
        self.features = SENetBackbone_tf(input_channels, input_size, in_plane=in_plane, num_block=num_block, name="features")
        self.aux_features = SENetBackbone_tf(num_aux, aux_input_size, in_plane=aux_in_plane, num_block=num_block, name="aux_features")
        
        self.log_scale = log_scale
        self.avgpool = layers.GlobalAveragePooling2D()
        self.relu = layers.ReLU()

        if input_size in [120, 160, 80]:
            num_reduce = int(math.floor(math.log(input_size / 2, 2)) - 3)
            aux_num_reduce = int(math.floor(math.log(aux_input_size / 2, 2)) - 3)
        elif input_size in [60, 40]:
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

        self.flatten = layers.Flatten()

        self.fc_height = layers.Dense(units=int(num_plane / 2), name="fc_height")
        self.bn_out_height = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name="bn_out_height")
        self.fc_out_height = layers.Dense(units=1, name="fc_out_height")

        self.fc_footprint = layers.Dense(units=int(num_plane / 2), name="fc_footprint")
        self.bn_out_footprint = layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name="bn_out_footprint")
        self.fc_out_footprint = layers.Dense(units=1, name="fc_out_footprint")
        
    def call(self, dta_input, training=None):
        x  = dta_input[:, :, :, :-1]
        aux = tf.expand_dims(dta_input[:, :, :, -1], axis=-1)

        x = self.features(x, training=training)
        x = self.avgpool(x)
        x = self.flatten(x)

        aux = self.aux_features(aux, training=training)
        aux = self.avgpool(aux)
        aux = self.flatten(aux)

        feature_shared = tf.concat([x, aux], axis=1)

        feature_fc1_footprint = self.fc_footprint(feature_shared)
        feature_fc1_footprint = self.bn_out_footprint(feature_fc1_footprint, training=training)
        feature_fc1_footprint = self.relu(feature_fc1_footprint)
        
        footprint = self.fc_out_footprint(feature_fc1_footprint)
        footprint = tf.keras.activations.sigmoid(footprint)

        feature_fc1_height = self.fc_height(feature_shared)
        feature_fc1_height = self.bn_out_height(feature_fc1_height, training=training)
        feature_fc1_height = self.relu(feature_fc1_height)

        if self.crossed:
            feature_fc1_height = feature_fc1_height * feature_fc1_footprint

        height = self.fc_out_height(feature_fc1_height)

        if not self.log_scale:
            height = self.relu(height)

        return footprint, height

    def get_para_shape(self):
        for l in self.layers:
            print([tensor.shape for tensor in l.get_weights()])

    def load_pretrained_model(self, trained_record, mode="torch"):
        if self.cuda_used:
            torch_state_dict = torch.load(trained_record)["state_dict"]
            for target_var in ["height", "footprint"]:
                # ------initialize fc layer
                tf_fc = self.get_layer("fc_{0}".format(target_var))
                weights = torch_state_dict["fc_{0}.weight".format(target_var)].cpu().numpy()
                bias = torch_state_dict["fc_{0}.bias".format(target_var)].cpu().numpy()
                tf_fc.set_weights([weights.transpose((1, 0)), bias])
                # ------initialize bn_out layer
                tf_bn_out = self.get_layer("bn_out_{0}".format(target_var))
                gamma = torch_state_dict["bn_out_{0}.weight".format(target_var)].cpu().numpy()
                beta = torch_state_dict["bn_out_{0}.bias".format(target_var)].cpu().numpy()
                mean = torch_state_dict["bn_out_{0}.running_mean".format(target_var)].cpu().numpy()
                var = torch_state_dict["bn_out_{0}.running_var".format(target_var)].cpu().numpy()
                tf_bn_out.set_weights([gamma, beta, mean, var])
                # ------initialize fc_out layer
                tf_fc_out = self.get_layer("fc_out_{0}".format(target_var))
                weights = torch_state_dict["fc_out_{0}.weight".format(target_var)].cpu().numpy()
                bias = torch_state_dict["fc_out_{0}.bias".format(target_var)].cpu().numpy()
                tf_fc_out.set_weights([weights.transpose((1, 0)), bias])
            # ------initialize backbone layer
            for bkbone in ["features", "aux_features"]:
                tf_backbone = self.get_layer(bkbone)
                # ---------[1] initialize the initial convolutional layer in the backbone layer
                tf_conv1 = tf_backbone.conv1
                weights = torch_state_dict[tf_conv1.name + ".weight"].cpu().numpy()
                weights = weights.transpose((2, 3, 1, 0))
                tf_conv1.set_weights([weights])
                # ---------[2] initialize the initial BatchNormalization layer in the backbone layer
                tf_bn1 = tf_backbone.bn1
                gamma = torch_state_dict[tf_bn1.name + ".weight"].cpu().numpy()
                beta = torch_state_dict[tf_bn1.name + ".bias"].cpu().numpy()
                mean = torch_state_dict[tf_bn1.name + ".running_mean"].cpu().numpy()
                var = torch_state_dict[tf_bn1.name + ".running_var"].cpu().numpy()
                tf_bn1.set_weights([gamma, beta, mean, var])
                # ---------[3] initialize the ResLayer in the backbone layer
                tf_res_layer = tf_backbone.res_layer
                # ---------------iteration over multiple res_layers
                for seq_layer in tf_res_layer.layers:
                    # ------------------iteration over multiple res_blocks in one res_layer
                    for res_block in seq_layer.layers:
                        # ---------------------iteration over components in one res_block
                        # ------------------------conv1
                        base_conv1 = res_block.conv1
                        weights = torch_state_dict[base_conv1.name + ".weight"].cpu().numpy()
                        weights = weights.transpose((2, 3, 1, 0))
                        base_conv1.set_weights([weights])
                        # ------------------------bn1
                        base_bn1 = res_block.bn1
                        gamma = torch_state_dict[base_bn1.name + ".weight"].cpu().numpy()
                        beta = torch_state_dict[base_bn1.name + ".bias"].cpu().numpy()
                        mean = torch_state_dict[base_bn1.name + ".running_mean"].cpu().numpy()
                        var = torch_state_dict[base_bn1.name + ".running_var"].cpu().numpy()
                        base_bn1.set_weights([gamma, beta, mean, var])
                        # ------------------------conv2
                        base_conv2 = res_block.conv2
                        weights = torch_state_dict[base_conv2.name + ".weight"].cpu().numpy()
                        weights = weights.transpose((2, 3, 1, 0))
                        base_conv2.set_weights([weights])
                        # ------------------------bn2
                        base_bn2 = res_block.bn2
                        gamma = torch_state_dict[base_bn2.name + ".weight"].cpu().numpy()
                        beta = torch_state_dict[base_bn2.name + ".bias"].cpu().numpy()
                        mean = torch_state_dict[base_bn2.name + ".running_mean"].cpu().numpy()
                        var = torch_state_dict[base_bn2.name + ".running_var"].cpu().numpy()
                        base_bn2.set_weights([gamma, beta, mean, var])
                        # ------------------------downsample
                        if res_block.downsample is not None:
                            base_downsample = res_block.downsample
                            for l in base_downsample.layers:
                                if "downsample.0" in l.name:
                                    base_conv = base_downsample.get_layer(l.name)
                                    weights = torch_state_dict[l.name + ".weight"].cpu().numpy()
                                    weights = weights.transpose((2, 3, 1, 0))
                                    base_conv.set_weights([weights])
                                elif "downsample.1" in l.name:
                                    base_bn = base_downsample.get_layer(l.name)
                                    gamma = torch_state_dict[l.name + ".weight"].cpu().numpy()
                                    beta = torch_state_dict[l.name + ".bias"].cpu().numpy()
                                    mean = torch_state_dict[l.name + ".running_mean"].cpu().numpy()
                                    var = torch_state_dict[l.name + ".running_var"].cpu().numpy()
                                    base_bn.set_weights([gamma, beta, mean, var])
                        # ------------------------SEfc
                        if hasattr(res_block, "SEfc"):
                            base_sefc = res_block.SEfc
                            for l in base_sefc.layers:
                                if "fc" in l.name:
                                    base_fc = base_sefc.get_layer(l.name)
                                    weights = torch_state_dict[l.name + ".weight"].cpu().numpy()
                                    base_fc.set_weights([weights.transpose((1, 0))])
        else:
            torch_state_dict = torch.load(trained_record, map_location=torch.device('cpu'))["state_dict"]
            for target_var in ["height", "footprint"]:
                # ------initialize fc layer
                tf_fc = self.get_layer("fc_{0}".format(target_var))
                weights = torch_state_dict["fc_{0}.weight".format(target_var)].numpy()
                bias = torch_state_dict["fc_{0}.bias".format(target_var)].numpy()
                tf_fc.set_weights([weights.transpose((1, 0)), bias])
                # ------initialize bn_out layer
                tf_bn_out = self.get_layer("bn_out_{0}".format(target_var))
                gamma = torch_state_dict["bn_out_{0}.weight".format(target_var)].numpy()
                beta = torch_state_dict["bn_out_{0}.bias".format(target_var)].numpy()
                mean = torch_state_dict["bn_out_{0}.running_mean".format(target_var)].numpy()
                var = torch_state_dict["bn_out_{0}.running_var".format(target_var)].numpy()
                tf_bn_out.set_weights([gamma, beta, mean, var])
                # ------initialize fc_out layer
                tf_fc_out = self.get_layer("fc_out_{0}".format(target_var))
                weights = torch_state_dict["fc_out_{0}.weight".format(target_var)].numpy()
                bias = torch_state_dict["fc_out_{0}.bias".format(target_var)].numpy()
                tf_fc_out.set_weights([weights.transpose((1, 0)), bias])
            # ------initialize backbone layer
            for bkbone in ["features", "aux_features"]:
                tf_backbone = self.get_layer(bkbone)
                # ---------[1] initialize the initial convolutional layer in the backbone layer
                tf_conv1 = tf_backbone.conv1
                weights = torch_state_dict[tf_conv1.name + ".weight"].numpy()
                weights = weights.transpose((2, 3, 1, 0))
                tf_conv1.set_weights([weights])
                # ---------[2] initialize the initial BatchNormalization layer in the backbone layer
                tf_bn1 = tf_backbone.bn1
                gamma = torch_state_dict[tf_bn1.name + ".weight"].numpy()
                beta = torch_state_dict[tf_bn1.name + ".bias"].numpy()
                mean = torch_state_dict[tf_bn1.name + ".running_mean"].numpy()
                var = torch_state_dict[tf_bn1.name + ".running_var"].numpy()
                tf_bn1.set_weights([gamma, beta, mean, var])
                # ---------[3] initialize the ResLayer in the backbone layer
                tf_res_layer = tf_backbone.res_layer
                # ---------------iteration over multiple res_layers
                for seq_layer in tf_res_layer.layers:
                    # ------------------iteration over multiple res_blocks in one res_layer
                    for res_block in seq_layer.layers:
                        # ---------------------iteration over components in one res_block
                        # ------------------------conv1
                        base_conv1 = res_block.conv1
                        weights = torch_state_dict[base_conv1.name + ".weight"].numpy()
                        weights = weights.transpose((2, 3, 1, 0))
                        base_conv1.set_weights([weights])
                        # ------------------------bn1
                        base_bn1 = res_block.bn1
                        gamma = torch_state_dict[base_bn1.name + ".weight"].numpy()
                        beta = torch_state_dict[base_bn1.name + ".bias"].numpy()
                        mean = torch_state_dict[base_bn1.name + ".running_mean"].numpy()
                        var = torch_state_dict[base_bn1.name + ".running_var"].numpy()
                        base_bn1.set_weights([gamma, beta, mean, var])
                        # ------------------------conv2
                        base_conv2 = res_block.conv2
                        weights = torch_state_dict[base_conv2.name + ".weight"].numpy()
                        weights = weights.transpose((2, 3, 1, 0))
                        base_conv2.set_weights([weights])
                        # ------------------------bn2
                        base_bn2 = res_block.bn2
                        gamma = torch_state_dict[base_bn2.name + ".weight"].numpy()
                        beta = torch_state_dict[base_bn2.name + ".bias"].numpy()
                        mean = torch_state_dict[base_bn2.name + ".running_mean"].numpy()
                        var = torch_state_dict[base_bn2.name + ".running_var"].numpy()
                        base_bn2.set_weights([gamma, beta, mean, var])
                        # ------------------------downsample
                        if res_block.downsample is not None:
                            base_downsample = res_block.downsample
                            for l in base_downsample.layers:
                                if "downsample.0" in l.name:
                                    base_conv = base_downsample.get_layer(l.name)
                                    weights = torch_state_dict[l.name + ".weight"].numpy()
                                    weights = weights.transpose((2, 3, 1, 0))
                                    base_conv.set_weights([weights])
                                elif "downsample.1" in l.name:
                                    base_bn = base_downsample.get_layer(l.name)
                                    gamma = torch_state_dict[l.name + ".weight"].numpy()
                                    beta = torch_state_dict[l.name + ".bias"].numpy()
                                    mean = torch_state_dict[l.name + ".running_mean"].numpy()
                                    var = torch_state_dict[l.name + ".running_var"].numpy()
                                    base_bn.set_weights([gamma, beta, mean, var])
                        # ------------------------SEfc
                        if hasattr(res_block, "SEfc"):
                            base_sefc = res_block.SEfc
                            for l in base_sefc.layers:
                                if "fc" in l.name:
                                    base_fc = base_sefc.get_layer(l.name)
                                    weights = torch_state_dict[l.name + ".weight"].numpy()
                                    base_fc.set_weights([weights.transpose((1, 0))])
        


def model_SEResNetAuxTF(target_resolution: int, log_scale=False, activation="relu", cuda_used=True, model_resaved=False, **kwargs) -> BuildingNet_aux_tf:
    psize = kernel_size_ref[target_resolution]
    if psize == 15:
        in_plane = 64
        num_block = 2
    elif psize == 30:
        in_plane = 64
        num_block = 1
    elif psize == 60:
        in_plane = 64
        num_block = 1
    else:
        in_plane = 64
        num_block = 1

    model = BuildingNet_aux_tf(input_channels=6, input_size=psize, aux_input_size=psize,
                                in_plane=in_plane, num_aux=1, num_block=num_block, log_scale=log_scale, activation=activation, cuda_used=cuda_used)
    model.build(input_shape=(None, psize, psize, 7))

    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
        if trained_record is not None:
            model.load_pretrained_model(trained_record)

    if model_resaved:
        if "saved_path_tf" in kwargs.keys():
            saved_path_tf = kwargs["saved_path_tf"]
        else:
            saved_path_torch = os.path.dirname(kwargs["trained_record"])
            saved_path_tf = saved_path_torch + "_TF"
        model.compute_output_shape(input_shape=(None, psize, psize, 7))
        model.save(saved_path_tf)
   
    total_num = sum([count_params(w) for w in model.trainable_weights]) + sum([count_params(w) for w in model.non_trainable_weights])
    trainable_num = sum([count_params(w) for w in model.trainable_weights])
    print("Total parameter of SEResNet: ", total_num, " Trainable parameter of SEResNet: ", trainable_num)
    return model


def model_SEResNetMTLAuxTF(target_resolution: int, crossed=False, log_scale=False, cuda_used=True, model_resaved=False, **kwargs) -> BuildingNet_aux_tf:
    psize = kernel_size_ref[target_resolution]
    if psize == 15:
        in_plane = 64
        num_block = 2
    elif psize == 30:
        in_plane = 64
        num_block = 1
    elif psize == 60:
        in_plane = 64
        num_block = 1
    else:
        in_plane = 64
        num_block = 1

    model = BuildingNetMTL_aux_tf(input_channels=6, input_size=psize, aux_input_size=psize,
                                    in_plane=in_plane, num_aux=1, num_block=num_block, crossed=crossed, log_scale=log_scale, cuda_used=cuda_used)
    model.build(input_shape=(None, psize, psize, 7))

    if "trained_record" in kwargs.keys():
        trained_record = kwargs["trained_record"]
        if trained_record is not None:
            model.load_pretrained_model(trained_record)
    
    if model_resaved:
        if "saved_path_tf" in kwargs.keys():
            saved_path_tf = kwargs["saved_path_tf"]
        else:
            saved_path_torch = os.path.dirname(kwargs["trained_record"])
            saved_path_tf = saved_path_torch + "_TF"
        # ------to load the model back, we can use `tf.keras.models.load_model`
        model.compute_output_shape(input_shape=(None, psize, psize, 7))
        model.save(saved_path_tf)
   
    total_num = sum([count_params(w) for w in model.trainable_weights]) + sum([count_params(w) for w in model.non_trainable_weights])
    trainable_num = sum([count_params(w) for w in model.trainable_weights])
    print("Total parameter of SEResNet: ", total_num, " Trainable parameter of SEResNet: ", trainable_num)
    return model


if __name__ == "__main__":
    '''
    # ---For the execution on MacOS, we may need to add the following two lines
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    '''

    # ---load pretrained weights from PyTorch
    pretrained_weight = os.path.join("dl-models", "height", "check_pt_senet_100m_MTL", "checkpoint.pth.tar")
    m = model_SEResNetMTLAuxTF(target_resolution=100, crossed=True, log_scale=False, trained_record=pretrained_weight, cuda_used=False, model_resaved=True, saved_path_tf="DL_run/height/check_pt_senet_100m_MTL_TF_gpu")
    
    # ---directly load pretrained weights from Tensorflow
    # pretrained_weight_tf = os.path.join("dl-models", "height", "check_pt_senet_100m_TF")
    # m = tf.keras.models.load_model(pretrained_weight_tf)
    '''
    # ---check the trainable_variables in Tensorflow's implementation
    a_list = []
    for v in m.trainable_variables:
        name = v.name
        name_parsed = name.split("/")[-2]
        a_list.append(name_parsed)

    a_list = np.unique(a_list)
    for a in a_list:
        print(a)
    '''

    # ---test whether the output of Tensorflow's implementation agrees with PyTorch's implementation
    test_dta = tf.concat([tf.ones(shape=[1, 20, 20, 7])*i*(-1.0) for i in range(0, 8)], axis=0)
    test_out = m(test_dta, training=False)
    print(test_out)
