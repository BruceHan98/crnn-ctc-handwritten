import torch
from utils.ctc_decoder import ctc_decode


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('batchnorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def cal_acc(log_probs, labels, lengths):
    with torch.no_grad():
        num_correct = 0

        preds = ctc_decode(log_probs, method='greedy', beam_size=10)
        gt_list = labels.cpu().numpy().tolist()
        lengths = lengths.cpu().numpy().tolist()

        len_counter = 0
        for pred, length in zip(preds, lengths):
            gt = gt_list[len_counter: len_counter + length]
            len_counter += length

            # print(pred, gt)
            for p, g in zip(pred, gt):
                if p == g:
                    num_correct += 1

        return float(num_correct / len_counter)


def adjust_learning_rate(optimizer, decay_rate):
    lr = optimizer.param_groups[0]['lr']
    if lr > 0.00001:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
        return optimizer.param_groups[0]['lr']
    else:
        return lr
