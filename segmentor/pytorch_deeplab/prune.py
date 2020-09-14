import torch
from torch.autograd import Variable
from torchvision import models
import numpy as np

def prune_xception_bn_layer(old_bn, filter_index):

    new_bn = \
        torch.nn.BatchNorm2d(old_bn.num_features-1)

    old_weights = old_bn.weight.data.cpu().numpy()
    new_weights = new_bn.weight.data.cpu().numpy()

    new_weights[: filter_index] = old_weights[: filter_index]
    new_weights[filter_index:] = old_weights[filter_index + 1:]

    if old_bn.bias is not None:
        bias_numpy = old_bn.bias.data.cpu().numpy()

        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index:] = bias_numpy[filter_index + 1:]
        new_bn.bias.data = torch.from_numpy(bias)

    return new_bn

def prune_xception_conv_layer(old_conv, filter_index, final_conv=False):


    if old_conv.groups == 1:
        if final_conv:
            new_conv = \
                torch.nn.Conv2d(in_channels=old_conv.in_channels - 1, \
                                out_channels=old_conv.out_channels, \
                                kernel_size=old_conv.kernel_size, \
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                dilation=old_conv.dilation,
                                groups=old_conv.groups,
                                bias=(old_conv.bias is not None))
        else:
            new_conv = \
                torch.nn.Conv2d(in_channels=old_conv.in_channels, \
                                out_channels=old_conv.out_channels - 1,
                                kernel_size=old_conv.kernel_size, \
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                dilation=old_conv.dilation,
                                groups=old_conv.groups,
                                bias=(old_conv.bias is not None))

    else:
        new_conv = \
            torch.nn.Conv2d(in_channels=old_conv.in_channels - 1, \
                            out_channels=old_conv.out_channels - 1,
                            kernel_size=old_conv.kernel_size, \
                            stride=old_conv.stride,
                            padding=old_conv.padding,
                            dilation=old_conv.dilation,
                            groups=old_conv.groups - 1,
                            bias=(old_conv.bias is not None))

    old_weights = old_conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    if final_conv:
        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
    else:
        new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
        new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]

    new_conv.weight.data = torch.from_numpy(new_weights)

    if old_conv.bias is not None:
        bias_numpy = old_conv.bias.data.cpu().numpy()

        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index:] = bias_numpy[filter_index + 1:]
        new_conv.bias.data = torch.from_numpy(bias)

    return new_conv

def update_original_model(model,layer,model_layer):

    if 'block' in model_layer:
        names = model_layer.split(".")
        block = names[2]

        if len(names) > 5: # in SeparableConv2d

            module = names[4][-2:-1]

            if isinstance(layer, torch.nn.modules.conv.Conv2d) and layer.groups > 1:
                model.backbone._modules[block].rep._modules[module].conv1 = layer
            elif isinstance(layer, torch.nn.modules.conv.Conv2d) and layer.groups == 1:
                model.backbone._modules[block].rep._modules[module].pointwise = layer
            else:
                model.backbone._modules[block].rep._modules[module].bn = layer
        elif 'skip' in model_layer:
            model.backbone._modules[block].skip = layer
        else:
            module = names[4][-2:-1]
            model.backbone._modules[block].rep._modules[module] = layer
    else:
        if 'SeparableConv2d' in str(layer):
            names = model_layer.split(".")
            if isinstance(layer, torch.nn.modules.conv.Conv2d) and layer.groups > 1:
                model.backbone._modules[names[-1]].conv1 = layer
            elif isinstance(layer, torch.nn.modules.conv.Conv2d) and layer.groups == 1:
                model.backbone._modules[names[-1]].pointwise = layer
            else:
                model.backbone._modules[names[-1]].bn = layer
        elif 'conv3' in model_layer or 'conv4' in model_layer or 'conv5' in model_layer:
            names = model_layer.split(".")
            model.backbone._modules[names[2]]._modules[names[3]] = layer
        else:
            names = model_layer.split(".")
            model.backbone._modules[names[-1]] = layer

    return model

# def do_prune_skip(model_layer, model):
#
#     prune = False
#     final_conv = False
#
#     if 'block' in model_layer:
#         names = model_layer.split(".")
#         block = names[2]
#         module = names[4][-2:-1]
#         layer = names[-1]
#
#         if module == '0' and layer == 'conv1':
#             prune = True
#             final_conv = True
#         elif module == str(len(model.backbone._modules[block].rep._modules)-1):
#             prune = True
#
#
#     return prune, final_conv

def prune_xception_layer(model, layer_index, filter_index, flat_backbone, model_dict):

    # layer_index = 58

    debug = False
    old_conv = flat_backbone[layer_index]
    final_conv_idx = None
    first_conv_idx = None
    # prune_skip = False
    offset = 1
    pruned_layers = []

    while layer_index + offset < len(flat_backbone):
        res = flat_backbone[layer_index + offset]
        if isinstance(res, torch.nn.modules.conv.Conv2d) and res.groups == 1:
            final_conv_idx = layer_index + offset
            break
        offset = offset + 1

    if old_conv.groups > 1: # need to also find previous conv
        offset = 1

        while layer_index - offset > -1:
            res = flat_backbone[layer_index - offset]
            if isinstance(res, torch.nn.modules.conv.Conv2d) and res.groups == 1:
                first_conv_idx = layer_index - offset
                break
            offset = offset + 1
    else:
        first_conv_idx = layer_index


    if final_conv_idx is None: # skip trying to prune the last layer in backbone
        return model, pruned_layers

    for i in range (first_conv_idx,final_conv_idx+1):
        model_layer = model_dict[i]
        if 'block' in model_layer:
            names = model_layer.split(".")
            block = names[2]
            module = names[4][-2:-1]
            layer = names[-1]
            if (module == '0' and layer == 'conv1') or module == str(len(model.backbone._modules[block].rep._modules) - 1):
                if debug:
                    print('skip prunning since it requires changing the block\'s input\output sizes')
                return model, pruned_layers

    for i in range (first_conv_idx,final_conv_idx+1):
        layer = flat_backbone[i]
        model_layer = model_dict[i]

        if debug:
            print('     pruned layer: {}'.format(model_layer))
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            flat_backbone[i] = prune_xception_conv_layer(layer, filter_index,final_conv = (i==final_conv_idx))
            model = update_original_model(model,flat_backbone[i],model_layer)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            flat_backbone[i] = prune_xception_bn_layer(layer, filter_index)
            model = update_original_model(model, flat_backbone[i], model_layer)
        # else:
        #     skip Relu layer

        pruned_layers.append(i)
        # prune_skip, final_conv = do_prune_skip(model_layer, model)
        #
        # if prune_skip:
        #     names = model_layer.split(".")
        #     block = names[2]
        #     layer = model.backbone._modules[block].skip
        #
        #     if layer is not None:
        #         prunned_layer = prune_xception_conv_layer(layer, filter_index,final_conv=final_conv)
        #         model = update_original_model(model, prunned_layer, 'model.backbone.{}.skip'.format(block))
        #         print('     pruned layer: model.backbone.{}.skip'.format(block))
        #     else:
        #         model.backbone._modules[block].prune_filters.append(filter_index)

    return model, pruned_layers

if __name__ == '__main__':
    model = models.vgg16(pretrained=True)
    model.train()

    t0 = time.time()
    model = prune_conv_layer(model, 28, 10)
    print("The prunning took", time.time() - t0)