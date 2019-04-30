import torch
from torch.autograd import Variable
import time
import os
import sys
import torch.nn.functional as F
from utils import AverageMeter, calculate_accuracy

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, tb_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    for i, (inputs, targets, anchor_inputs, positive_inputs, negative_inputs, negative_targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        itr = (epoch - 1) * len(data_loader) + (i + 1)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)

        # ----------------------------------------------
        # margin = 1.0
        margin = 0.5
        _, anchor_features = model(anchor_inputs)
        _, positive_features = model(positive_inputs)
        _, negative_features = model(negative_inputs)
        anchor_features_normalized= (anchor_features/(torch.norm(anchor_features,p=2,dim=1).unsqueeze(1)))
        positive_features_normalized=  (positive_features/(torch.norm(positive_features,p=2,dim=1).unsqueeze(1)))
        negative_features_normalized=  (negative_features/(torch.norm(negative_features,p=2,dim=1).unsqueeze(1)))
        distance_positive = (anchor_features_normalized - positive_features_normalized).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor_features_normalized - negative_features_normalized).pow(2).sum(1)  # .pow(.5)
        triplet_losses = F.relu(distance_positive - distance_negative + margin)
        triplet_loss = triplet_losses.mean()
        
        loss = triplet_loss
        tb_logger.add_scalar('train/triplet_loss', loss, itr)
        # ----------------------------------------------
        # outputs, features = model(anchor_inputs)
        # loss = criterion(outputs, targets)
        # acc = calculate_accuracy(outputs, targets)
        acc = 0.0

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        tb_logger.add_scalar('train/acc', accuracies.val, itr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': itr,
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
