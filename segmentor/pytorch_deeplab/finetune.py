import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from modeling.deeplab import *
from dataloaders import make_data_loader
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
from utils.metrics import Evaluator

class FilterPrunner:
    def __init__(self, model,use_cuda):
        self.model = model
        self.reset()
        self.use_cuda = use_cuda
        self.flat_backbone , self.separable_idx , self.blocks_idx, self.model_dict = self.create_flat_model()

    def create_flat_model(self):

        def flatten_block(block, block_idx=0):

            for (name, module) in block._modules.items():

                if 'SeparableConv2d' in str(module):  # if sequential layer, apply recursively to layers in sequential layer
                    # print('@@@ start flattening SeparableConv2d')
                    separable.append(len(all_layers))
                    # print('[{}] - {}'.format(len(all_layers), 'fixed_padding'))
                    layers_dict[len(all_layers)] = 'model.backbone.block{}.rep._modules[{}].{}'.format(str(block_idx), name,'conv1')
                    all_layers.append(module.conv1)
                    layers_dict[len(all_layers)] = 'model.backbone.block{}.rep._modules[{}].{}'.format(str(block_idx), name,'bn')
                    all_layers.append(module.bn)
                    layers_dict[len(all_layers)] = 'model.backbone.block{}.rep._modules[{}].{}'.format(str(block_idx), name,'pointwise')
                    all_layers.append(module.pointwise)
                    # print('@@@ finished flattening SeparableConv2d')

                else:
                    layers_dict[len(all_layers)] = 'model.backbone.block{}.rep._modules[{}]'.format(str(block_idx), name)
                    all_layers.append(module)



        all_layers = []
        separable = []
        blocks = [[] for i in range(20)]
        layers_dict = {}

        # Entry flow
        layers_dict[len(all_layers)] = 'model.backbone.conv1'
        all_layers.append(self.model.backbone.conv1)
        layers_dict[len(all_layers)] = 'model.backbone.bn1'
        all_layers.append(self.model.backbone.bn1)
        layers_dict[len(all_layers)] = 'model.backbone.relu'
        all_layers.append(self.model.backbone.relu)

        layers_dict[len(all_layers)] = 'model.backbone.conv2'
        all_layers.append(self.model.backbone.conv2)
        layers_dict[len(all_layers)] = 'model.backbone.bn2'
        all_layers.append(self.model.backbone.bn2)
        layers_dict[len(all_layers)] = 'model.backbone.relu'
        all_layers.append(self.model.backbone.relu)


        blocks[0].append(len(all_layers))
        flatten_block(self.model.backbone.block1.rep,block_idx=1)
        blocks[0].append(len(all_layers))

        layers_dict[len(all_layers)] = 'model.backbone.relu'
        all_layers.append(self.model.backbone.relu)

        blocks[1].append(len(all_layers))
        flatten_block(self.model.backbone.block2.rep,block_idx=2)
        blocks[1].append(len(all_layers))

        blocks[2].append(len(all_layers))
        flatten_block(self.model.backbone.block3.rep,block_idx=3)
        blocks[2].append(len(all_layers))

        # Middle flow
        blocks[3].append(len(all_layers))
        flatten_block(self.model.backbone.block4.rep, block_idx=4)
        blocks[3].append(len(all_layers))

        blocks[4].append(len(all_layers))
        flatten_block(self.model.backbone.block5.rep, block_idx=5)
        blocks[4].append(len(all_layers))

        blocks[5].append(len(all_layers))
        flatten_block(self.model.backbone.block6.rep, block_idx=6)
        blocks[5].append(len(all_layers))

        blocks[6].append(len(all_layers))
        flatten_block(self.model.backbone.block7.rep, block_idx=7)
        blocks[6].append(len(all_layers))

        blocks[7].append(len(all_layers))
        flatten_block(self.model.backbone.block8.rep, block_idx=8)
        blocks[7].append(len(all_layers))

        blocks[8].append(len(all_layers))
        flatten_block(self.model.backbone.block9.rep, block_idx=9)
        blocks[8].append(len(all_layers))

        blocks[9].append(len(all_layers))
        flatten_block(self.model.backbone.block10.rep, block_idx=10)
        blocks[9].append(len(all_layers))

        blocks[10].append(len(all_layers))
        flatten_block(self.model.backbone.block11.rep, block_idx=11)
        blocks[10].append(len(all_layers))

        blocks[11].append(len(all_layers))
        flatten_block(self.model.backbone.block12.rep, block_idx=12)
        blocks[11].append(len(all_layers))

        blocks[12].append(len(all_layers))
        flatten_block(self.model.backbone.block13.rep, block_idx=13)
        blocks[12].append(len(all_layers))

        blocks[13].append(len(all_layers))
        flatten_block(self.model.backbone.block14.rep, block_idx=14)
        blocks[13].append(len(all_layers))

        blocks[14].append(len(all_layers))
        flatten_block(self.model.backbone.block15.rep, block_idx=15)
        blocks[14].append(len(all_layers))

        blocks[15].append(len(all_layers))
        flatten_block(self.model.backbone.block16.rep, block_idx=16)
        blocks[15].append(len(all_layers))

        blocks[16].append(len(all_layers))
        flatten_block(self.model.backbone.block17.rep, block_idx=17)
        blocks[16].append(len(all_layers))

        blocks[17].append(len(all_layers))
        flatten_block(self.model.backbone.block18.rep, block_idx=18)
        blocks[17].append(len(all_layers))

        blocks[18].append(len(all_layers))
        flatten_block(self.model.backbone.block19.rep, block_idx=19)
        blocks[18].append(len(all_layers))

        # Exit flow
        blocks[19].append(len(all_layers))
        flatten_block(self.model.backbone.block20.rep, block_idx=20)
        blocks[19].append(len(all_layers))

        layers_dict[len(all_layers)] = 'model.backbone.relu'
        all_layers.append(self.model.backbone.relu)
        separable.append(len(all_layers))

        # print('@@@ start flattening SeparableConv2d')
        separable.append(len(all_layers))
        # print('[{}] - {}'.format(len(all_layers), 'fixed_padding'))

        layers_dict[len(all_layers)] = 'model.backbone.conv3.conv1'
        all_layers.append(self.model.backbone.conv3.conv1)
        layers_dict[len(all_layers)] = 'model.backbone.conv3.bn'
        all_layers.append(self.model.backbone.conv3.bn)
        layers_dict[len(all_layers)] = 'model.backbone.conv3.pointwise'
        all_layers.append(self.model.backbone.conv3.pointwise)

        # print('@@@ finished flattening SeparableConv2d')

        layers_dict[len(all_layers)] = 'model.backbone.bn3'
        all_layers.append(self.model.backbone.bn3)
        layers_dict[len(all_layers)] = 'model.backbone.relu'
        all_layers.append(self.model.backbone.relu)

        # print('@@@ start flattening SeparableConv2d')
        separable.append(len(all_layers))
        # print('[{}] - {}'.format(len(all_layers), 'fixed_padding'))

        layers_dict[len(all_layers)] = 'model.backbone.conv4.conv1'
        all_layers.append(self.model.backbone.conv4.conv1)
        layers_dict[len(all_layers)] = 'model.backbone.conv4.bn'
        all_layers.append(self.model.backbone.conv4.bn)
        layers_dict[len(all_layers)] = 'model.backbone.conv4.pointwise'
        all_layers.append(self.model.backbone.conv4.pointwise)

        # print('@@@ finished flattening SeparableConv2d')

        layers_dict[len(all_layers)] = 'model.backbone.bn4'
        all_layers.append(self.model.backbone.bn4)
        layers_dict[len(all_layers)] = 'model.backbone.relu'
        all_layers.append(self.model.backbone.relu)

        # print('@@@ start flattening SeparableConv2d')
        separable.append(len(all_layers))
        # print('[{}] - {}'.format(len(all_layers), 'fixed_padding'))

        layers_dict[len(all_layers)] = 'model.backbone.conv5.conv1'
        all_layers.append(self.model.backbone.conv5.conv1)
        layers_dict[len(all_layers)] = 'model.backbone.conv5.bn'
        all_layers.append(self.model.backbone.conv5.bn)
        layers_dict[len(all_layers)] = 'model.backbone.conv5.pointwise'
        all_layers.append(self.model.backbone.conv5.pointwise)

        # print('@@@ finished flattening SeparableConv2d')

        layers_dict[len(all_layers)] = 'model.backbone.bn5'
        all_layers.append(self.model.backbone.bn5)
        layers_dict[len(all_layers)] = 'model.backbone.relu'
        all_layers.append(self.model.backbone.relu)

        # for i in range(len(all_layers)):
        #     print('[{}] - {}'.format(i,all_layers[i]))
        #
        # for i in range(len(all_layers)):
        #     print('[{}] - {}'.format(i,layers_dict[i]))


        return all_layers,separable,blocks,layers_dict


    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        input = x

        blocks_dic = {
            0: self.model.backbone.block1,
            1: self.model.backbone.block2,
            2: self.model.backbone.block3,
            3: self.model.backbone.block4,
            4: self.model.backbone.block5,
            5: self.model.backbone.block6,
            6: self.model.backbone.block7,
            7: self.model.backbone.block8,
            8: self.model.backbone.block9,
            9: self.model.backbone.block10,
            10: self.model.backbone.block11,
            11: self.model.backbone.block12,
            12: self.model.backbone.block13,
            13: self.model.backbone.block14,
            14: self.model.backbone.block15,
            15: self.model.backbone.block16,
            16: self.model.backbone.block17,
            17: self.model.backbone.block18,
            18: self.model.backbone.block19,
            19: self.model.backbone.block20,
        }


        def fixed_padding(inputs, kernel_size, dilation):
            kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
            return padded_inputs


        activation_index = 0

        # Entry flow
        for i in range(6):
            module = self.flat_backbone[i]
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = i
                activation_index += 1

        # first block

        temp_input = x

        for layer in range (self.blocks_idx[0][0],self.blocks_idx[0][1]):

            module = self.flat_backbone[layer]
            if layer in self.separable_idx:
                x = fixed_padding(x, module.kernel_size[0], module.dilation[0])

            x = module(x)

            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        block = blocks_dic[0]

        if block.skip is not None:
            skip = block.skip(temp_input)
            skip = block.skipbn(skip)
        else:
            skip = temp_input

        x = x + skip

        module = self.flat_backbone[19]
        x = module(x)
        low_level_feat = x

        # blocks 2 - 20

        for block_idx in range (1,20):

            temp_input = x

            for layer in range(self.blocks_idx[block_idx][0], self.blocks_idx[block_idx][1]):

                module = self.flat_backbone[layer]

                if layer in self.separable_idx:
                    x = fixed_padding(x, module.kernel_size[0], module.dilation[0])

                x = module(x)

                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    x.register_hook(self.compute_rank)
                    self.activations.append(x)
                    self.activation_to_layer[activation_index] = layer
                    activation_index += 1

            block = blocks_dic[block_idx]

            if block.skip is not None:
                skip = block.skip(temp_input)
                skip = block.skipbn(skip)
            else:
                skip = temp_input

            x = x + skip

        # last layers after blocks
        for i in range(self.blocks_idx[block_idx][1]+1,len(self.flat_backbone)):

            module = self.flat_backbone[i]

            if layer in self.separable_idx:
                x = fixed_padding(x, module.kernel_size[0], module.dilation[0])

            x = module(x)

            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = i
                activation_index += 1

        x = self.model.aspp(x)
        x = self.model.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation.float() * grad.float()
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data


        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / torch.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            # for i in range(len(filters_to_prune_per_layer[l])):
            #     filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune             

class PrunningFineTuner_DEEPLAB:
    def __init__(self, backbone, num_classes, train_data_loader, test_data_loader, use_cuda , model):

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.prunner = FilterPrunner(self.model,use_cuda)
        self.model.train()
        self.evaluator = Evaluator(num_classes)
        self.use_cuda = use_cuda



    def test(self):
        # return
        print('Testing model:')
        self.model.eval()
        self.evaluator.reset()
        mean_infer_time = []


        for i, sample in enumerate(self.test_data_loader):
            if i % 50 == 0:
                print('Processing batch {}/{}'.format(i,int(len(self.test_data_loader))))
            t0 = time.time()
            batch, target = sample['image'], sample['label']
            if self.use_cuda:
                batch = batch.cuda()
            input = Variable(batch)
            output = self.model(input)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            t_total = (time.time() - t0)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            mean_infer_time.append(t_total)

        m_time = np.mean(mean_infer_time)
        print("Mean inference after pruning took {} ms per epoch.".format(m_time*1000))

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print("Acc:{0:.3f}, Acc_class:{1:.3f}, mIoU:{2:.3f}, fwIoU: {3:.3f}".format(Acc*100, Acc_class*100, mIoU*100, FWIoU*100))

        self.model.train()

        return m_time, Acc, Acc_class, mIoU, FWIoU

    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(self.model.classifier.parameters(), lr=0.0001, momentum=0.9)


        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            # self.test()
        print("Finished fine tuning.")

        return self.test()

    def train_batch(self, optimizer, batch, label, rank_filters):

        if self.use_cuda:
            batch = batch.cuda()
            label = label.cuda()

        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            loss = self.criterion(output, label.long())
            loss.backward()
        else:
            output = self.model(input)
            loss = self.criterion(output, label.long())
            loss.backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, rank_filters=False):

        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)

        for i, sample in enumerate(self.train_data_loader):
            if i % 100 == 0:
                print('Processing batch {}/{}'.format(i,int(len(self.train_data_loader))))
            # if i>20:
            #     break;
            batch, label = sample['image'], sample['label']
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters=True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):

        filters = 0

        def count_conv_layers(network,filters=0):

            for layer in network.children():
                # print(type(layer))
                if isinstance(layer, torch.nn.modules.conv.Conv2d):
                    filters = filters + layer.out_channels
                elif 'Block' in str(type(layer)):  # if sequential layer, apply recursively to layers in sequential layer
                    filters += count_conv_layers(layer)
                elif 'SeparableConv2d' in str(type(layer)):  # if sequential layer, apply recursively to layers in sequential layer
                    filters += count_conv_layers(layer)
                elif type(layer) == nn.Sequential:
                    filters += count_conv_layers(layer)

            return filters

        # for name, module in self.model.backbone._modules.items():
        #     if isinstance(module, torch.nn.modules.conv.Conv2d):
        #         filters = filters + module.out_channels
        #     elif 'block' in name:
        #         for block_name, block_module in module._modules.items():
        #             if isinstance(block_module, torch.nn.modules.conv.Conv2d):
        #                 filters = filters + block_module.out_channels
        return count_conv_layers(self.model.backbone,filters)

    def prune(self):
        # Get the accuracy before prunning
        self.test()
        self.model.train()

        epoch_times = []
        # Make sure all the layers are trainable
        for param in self.model.backbone.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 256
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * 2.0 / 4)

        print("Number of prunning iterations to reduce 50% filters", iterations)

        for n in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()

            skip = []
            for i, (layer_index, filter_index) in enumerate(prune_targets):
                print('[{}] - Pruning layer {} and filter_index {}'.format(i,layer_index,filter_index))
                if i in skip or filter_index < 0:
                    print('skipped pruning layer ', i)
                    continue

                model, update_pruned_layers = prune_xception_layer(model, layer_index, filter_index, self.prunner.flat_backbone, self.prunner.model_dict)

                # fix filters' indices
                for l, (l_index, f_index) in enumerate(prune_targets):
                    if l_index in update_pruned_layers and f_index >= filter_index:
                        if f_index == filter_index:
                            skip.append(l_index)
                        prune_targets[l] = (l_index, f_index-1)


            self.model = model
            if self.use_cuda:
                self.model = self.model.cuda()

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
            m_time, Acc, Acc_class, mIoU, FWIoU = self.train(optimizer, epoches=10)
            epoch_times.append(m_time)
            # self.test()

            torch.save(self.model, "/home/ido/Deep/Pytorch/pytorch_deeplab-xception/pruned_models/iter:{0}_time:{1:.2f}_Acc:{2:.3f}_Acc_class:{3:.3f}_mIoU:{4:.3f}_fwIoU:{5:.3f}.pth".format(n+221,m_time*1000,Acc*100, Acc_class*100, mIoU*100, FWIoU*100))

        print(epoch_times)
        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches=15)
        torch.save(model.state_dict(), "model_prunned")

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus pruning")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'arbel'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether to use balanced weights for training (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    # Define Dataloader
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)

    model = DeepLab(num_classes=nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        # checkpoint = torch.load(args.resume)
        model = torch.load(args.resume)
        # args.start_epoch = checkpoint['epoch']
        # model.load_state_dict(checkpoint['state_dict'])
        # best_pred = checkpoint['best_pred']
        # print("=> loaded checkpoint '{}' (epoch {})"
        #       .format(args.resume, checkpoint['epoch']))

    else:
        print('must supply a trained model.')
        return

    if args.cuda:
        model = model.cuda()

    fine_tuner = PrunningFineTuner_DEEPLAB(args.backbone, nclass, train_loader, val_loader, args.cuda, model)

    # if args.train:
    #     fine_tuner.train(epoches=10)
    #     torch.save(model, "model")

    fine_tuner.prune()

if __name__ == "__main__":
   main()





