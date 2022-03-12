import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.backends import cudnn
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

from config import args
from dataset import SynthHWCL, transformer, collate_fn
from model.crnn import CRNN
from utils.ctc_decoder import ctc_decode
from utils.logger import logger
from utils.train_utils import weights_init, cal_acc

# setup
random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

cudnn.benchmark = True
gpu_id = args.gpu_id
device = torch.device(f'cuda:{gpu_id}' if args.cuda else 'cpu')
logger.info("device: %s" % device)

# dataloader
train_dataset = SynthHWCL(args.train_data_dir, label_files=args.train_label, transform=transformer)
valid_dataset = SynthHWCL(args.valid_data_dir, label_files=args.valid_label, transform=None)
logger.info("train dataset length: %d" % len(train_dataset))
logger.info("valid dataset length: %d" % len(valid_dataset))

train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_workers,
                          collate_fn=collate_fn,
                          drop_last=True
                          )
valid_loader = DataLoader(valid_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_workers,
                          collate_fn=collate_fn,
                          drop_last=True
                          )


def train():
    model = CRNN(args.img_H, args.channel, args.num_classes, args.n_hidden)
    model.apply(weights_init)
    # logger.info(model)

    if args.pretrained_model != '' and os.path.exists(args.pretrained_model):
        logger.info('Loading pretrained model from %s' % args.pretrained_model)
        model.load_state_dict(torch.load(args.pretrained_model))
    else:
        logger.info('Training from scratch...')

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = CTCLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.8, last_epoch=-1)

    model.to(device)

    best_accuracy = 0.
    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
        train_loss = 0.
        accuracy = 0.
        cur_lr = None
        model.train()

        start_time = time.time()
        for step, (image_batch, label_batch, length_batch) in enumerate(train_loader):
            image_batch = image_batch.to(device)
            label_batch = label_batch
            length_batch = length_batch

            logits = model(image_batch)
            probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = image_batch.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
            target_lengths = torch.flatten(length_batch)

            loss = criterion(probs, label_batch, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # gradient clipping with 5
            optimizer.step()

            train_loss += loss
            accuracy += cal_acc(probs, label_batch, length_batch)

            # print log
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            if (step + 1) % args.print_interval == 0:
                logger.info('epoch: [%d/%d], iters: %d/%d, time: %.2fs, lr: %.6f, loss: %.6f, acc: %.4f' %
                      (epoch+1, args.epochs, step+1, len(train_loader), time.time() - start_time,
                       cur_lr, train_loss/args.print_interval, accuracy/args.print_interval))
                train_loss = 0.
                accuracy = 0.
                start_time = time.time()
        
        # adjust lr
        if cur_lr >= args.lr_min:
            lr_scheduler.step()

        # valid
        if (epoch + 1) % args.valid_interval == 0:
            valid_acc = valid(model, criterion, epoch + 1)
            if valid_acc > best_accuracy:
                best_model_path = os.path.join(args.save_model_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                best_accuracy = valid_acc
        
        # save model
        if epoch > 8:
            model_save_path = os.path.join(args.save_model_dir, time.strftime(f'%Y%m%d%H%M_epoch_{epoch+1}.pth'))
            torch.save(model.state_dict(), model_save_path)


def valid(model, criterion, epoch):
    logger.info("Start validation ...")
    model.eval()

    total_correct = 0
    total_num = 0
    with torch.no_grad():
        for step, (images, labels, lengths) in enumerate(valid_loader):
            if step == args.valid_iters:
                break

            images = images.to(device)
            labels = labels
            lengths = lengths

            logits = model(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, labels, input_lengths, lengths)

            preds = ctc_decode(log_probs, method='greedy')
            reals = labels.cpu().numpy().tolist()
            lengths = lengths.cpu().numpy().tolist()

            correct = 0
            target_length_counter = 0
            for pred, target_length in zip(preds, lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                if pred == real:
                    correct += 1
            logger.info("iter: %d/%d, loss: %.4f, acc: %.4f" %
                  (step+1, min(args.valid_iters, len(valid_loader)), loss.item(), correct/batch_size))

            total_correct += correct
            total_num += batch_size

        valid_acc = total_correct / total_num
        logger.info(f"Epoch {epoch}, valid acc: {valid_acc:.4f}")

        return valid_acc


if __name__ == '__main__':
    train()
