import os
import sys

import torch
import torch.utils.data
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from torch.backends import cudnn

from dataset import SynthHWCL, collate_fn
from config import args
from model.crnn import CRNN
from utils.ctc_decoder import ctc_decode
from utils.char_utils import get_char_dict, id2char

cudnn.benchmark = True
gpu_id = args.gpu_id
device = torch.device(f'cuda:{gpu_id}' if args.cuda else 'cpu')
print("device: %s" % device)

eval_dataset = SynthHWCL(args.eval_data_dir, label_files=args.eval_label, transform=None)
eval_loader = DataLoader(eval_dataset,
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=args.num_workers,
                         collate_fn=collate_fn,
                         drop_last=True
                         )
char_dict = get_char_dict(args.char_dict_path)


def Levenshtein_Distance(str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


def evaluate():
    model = CRNN(args.img_H, args.channel, args.num_classes, args.n_hidden)
    if args.pretrained_model != '' and os.path.exists(args.pretrained_model):
        print('Loading pretrained model from %s' % args.pretrained_model)
        model.load_state_dict(torch.load(args.pretrained_model))
    else:
        print('Model not exists!')
        sys.exit(1)

    criterion = CTCLoss(reduction='sum')
    model.to(device)

    print("Start validation ...")
    predict_txt = open('eval_result.txt', 'w', encoding='utf-8')
    total_correct = 0
    total_num = 0
    model.eval()
    with torch.no_grad():
        for step, (images, labels, lengths) in enumerate(eval_loader):
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

            dist_acc = 0.
            correct = 0
            target_length_counter = 0
            for pred, target_length in zip(preds, lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length

                gt = ''.join([id2char(r, char_dict) for r in real])
                predict = ''.join([id2char(p, char_dict) for p in pred])

                # acc
                if pred == real:
                    correct += 1

                # distance
                distance = Levenshtein_Distance(gt, predict)
                acc = distance / len(gt)
                dist_acc += acc

                # print(f'{gt} ==>{pred}')
                predict_txt.write(gt + '\t' + predict + '\t' + str(acc) + '\n')

            print("iter: %d/%d, loss: %.4f, acc: %.4f, dist_acc: %.4f" %
                  (step + 1, len(eval_loader), loss.item(), correct / batch_size, dist_acc / batch_size))
            total_correct += correct
            total_num += batch_size

        print(f"eval acc: {total_correct / total_num:.4f}")


if __name__ == '__main__':
    evaluate()
