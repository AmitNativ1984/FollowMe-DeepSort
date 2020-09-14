import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from finetune import FilterPrunner
from prune import prune_xception_layer
from torch.autograd import Variable
import time

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

from trains import Task, Logger
task = Task.init(project_name="BBOX segmentation - Pytorch",
                         task_name="pytorch deeplab - 256",
                         reuse_last_task_id=False,
                         continue_last_task=True)

class PrunningFineTuner_DEEPLAB:
    def __init__(self, train_data_loader, use_cuda , model, optimizer):

        self.train_data_loader = train_data_loader
        self.model = model
        # self.optimizer = optimizer
        self.prunner = FilterPrunner(self.model,use_cuda)
        self.model.train()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.use_cuda = use_cuda
        self.number_of_filters = self.total_num_filters()

    def train_epoch(self):

        for i, sample in enumerate(self.train_data_loader):
            if i % 100 == 0:
                print('Processing batch {}/{}'.format(i,int(len(self.train_data_loader))))
            # if i > 20:
            #     break
            input, label = sample['image'], sample['label']

            if self.use_cuda:
                input = input.cuda()
                label = label.cuda()

            self.model.zero_grad()
            # input = Variable(batch)

            output = self.prunner.forward(input)
            loss = self.criterion(output, label.long())
            loss.backward()

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch()
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

        return count_conv_layers(self.model.backbone,filters)

    def prune(self,num_filters_to_prune_per_iteration, model):

        self.model = model
        self.model.train()

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
        for i, (layer_index, filter_index) in enumerate(prune_targets):
            # print('[{}] - Pruning layer {} and filter_index {}'.format(i,layer_index,filter_index))
            model, update_pruned_layers = prune_xception_layer(model, layer_index, filter_index, self.prunner.flat_backbone, self.prunner.model_dict)

            # fix filters' indices
            for l, (l_index, f_index) in enumerate(prune_targets):
                if l_index in update_pruned_layers and f_index >= filter_index:
                    prune_targets[l] = (l_index, f_index-1)


        self.model = model
        if self.use_cuda:
            self.model = self.model.cuda()

        message = str(100 * float(self.total_num_filters()) / self.number_of_filters) + "%"
        print("Filters prunned", str(message))

        return self.model


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.use_amp = True if APEX_AVAILABLE and args.use_amp else False
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # optimizer = torch.optim.AdamW(train_params, weight_decay=args.weight_decay)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight[10] *= 10 # person
            weight[11] *= 10 # vehicle
            weight = torch.from_numpy(weight.astype(np.float32))

        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader),lr_step = args.lr_step)

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()

        # mixed precision
        if self.use_amp and args.cuda:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level="O1",
                keep_batchnorm_fp32=None, loss_scale="dynamic", max_loss_scale=2
            )

        # Using data parallel
        if args.cuda and len(args.gpu_ids)>1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            # self.model = apex.parallel.DistributedDataParallel(self.model)

        # self.optimizer = FP16_Optimizer(self.optimizer,dynamic_loss_scale=True)
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda and len(args.gpu_ids)>1:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % (train_loss/len(self.train_loader)))

        # if self.args.no_val:
            # save checkpoint every epoch
        is_best = False
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict() ,
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best, filename='checkpoint_fp16_02_os8.pth')


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(image)

            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.test_batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % (test_loss/len(self.val_loader)))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'arbel', 'bbox'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze_bn', type=bool, default=False,
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
    parser.add_argument('--lr_step', type=int, default=50,
                        help='number of steps before lr reduction (default: 100)')
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
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='whether to use mixed precision (default: False)')
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
    parser.add_argument('--prune', action='store_true', default=False,
                        help='perform pruning iterations')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    if args.prune:

        fine_tuner = PrunningFineTuner_DEEPLAB(trainer.train_loader, args.cuda, trainer.model.module,trainer.optimizer)

        # Make sure all the layers are trainable
        for param in fine_tuner.model.backbone.parameters():
            param.requires_grad = True

        number_of_filters = fine_tuner.total_num_filters()
        num_filters_to_prune_per_iteration = 256
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * 2.0 / 4)

        print("Number of prunning iterations to reduce 50% filters", iterations)

        t0 = time.time()
        trainer.validation(0)
        infer_time = time.time() - t0
        print("Time of validation of original model is {} seconds.".format(infer_time))

        for n in range(iterations):

            trainer.model.module = fine_tuner.prune(num_filters_to_prune_per_iteration,trainer.model.module)
            print("Fine tuning to recover from prunning iteration.")
            for epoch in range(10):
                trainer.training(epoch)

            t0 = time.time()
            trainer.validation(0)
            infer_time = time.time() - t0
            print("Time of validation after pruning iteration {} is {} seconds.".format(n,infer_time))
            torch.save(trainer.model.module, "/home/ido/Deep/Pytorch/pytorch_deeplab-xception/pruned_models/iter_{}_time_{}.pth".format(n,infer_time))

        print("Finished. Going to fine tune the model a bit more")
        for epoch in range(15):
            trainer.training(epoch)

        t0 = time.time()
        trainer.validation(0)
        infer_time = time.time() - t0
        print("Time of validation of final prunned model is {} seconds.".format(infer_time))
        torch.save(model.state_dict(), "model_prunned")
    else:
        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
                trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
