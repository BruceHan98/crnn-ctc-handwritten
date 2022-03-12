import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from config import args
from model.crnn import CRNN
from utils.char_utils import get_char_dict, id2char
from utils.ctc_decoder import ctc_decode


gpu_id = args.gpu_id
device = torch.device(f'cuda:{gpu_id}' if args.cuda else 'cpu')
print("device: %s" % device)

model = CRNN(args.img_H, args.channel, args.num_classes, args.n_hidden)
print('loading pretrained model from', args.pretrained_model)
st = time.time()
model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
print("cost time: %.4fs" % (time.time() - st))

image_dir = args.test_dir
image_paths = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp']:
            image_paths.append(os.path.join(root, file))
char_dict = get_char_dict(args.char_dict_path)


def predict():
    model.eval()
    with torch.no_grad():
        for image_path in image_paths:
            start_time = time.time()
            image = Image.open(image_path).convert("L")
            w, h = image.size
            ratio = args.img_H / h
            image = image.resize((int(ratio * w), args.img_H), Image.ANTIALIAS)
            w, h = image.size

            # image = np.array(image)
            # image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
            # cv2.imshow('test', image)
            # cv2.waitKey(0)

            image = np.array(image).reshape((1, 1, h, w))
            image = (image / 127.5) - 1.0
            image = torch.FloatTensor(image)

            logits = model(image)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            pred = ctc_decode(log_probs, method='greedy', beam_size=10)
            pred = ''.join([id2char(p, char_dict) for p in pred[0]])
            print(f"{image_path}: {pred}, {time.time() - start_time:.4f}s")


if __name__ == '__main__':
    predict()
