import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--gpu_id", type=str, default="7")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=256)

parser.add_argument("--save_model_dir", type=str, default='outputs')
parser.add_argument("--train_data_dir", type=str, default='../hwrec_train')
parser.add_argument("--valid_data_dir", type=str, default='../hwrec_test')
parser.add_argument("--eval_data_dir", type=str, default='../hwrec_test')
parser.add_argument("--train_label", type=list, default=['../hwrec_train/hwrec_gt_train.txt'], help='path of train label file')
parser.add_argument("--valid_label", type=list, default=['../hwrec_test/hwrec_gt_test.txt'], help='path of valid label file')
parser.add_argument("--eval_label", type=list, default=['../hwrec_test/hwrec_gt_test.txt'], help='path of evaluate label file')
parser.add_argument("--char_dict_path", type=str, default='hw_chars.txt', help='path of char dict')
parser.add_argument("--pretrained_model", "-ptr", type=str, default='')

parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--start_epoch", type=int, default=0, help="recovery from break")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--lr_min", type=float, default=1e-5)
parser.add_argument("--print_interval", type=int, default=10, help="print logs per interval iters")
parser.add_argument("--valid_interval", type=int, default=1, help="valid model per interval epochs")
parser.add_argument("--valid_iters", type=int, default=20, help="valid iters")

parser.add_argument("--img_H", type=int, default=48)
parser.add_argument("--img_W", type=int, default=600)
parser.add_argument("--channel", type=int, default=1, help='channel of input image')
parser.add_argument("--num_classes", type=int, default=7356, help='num of characters')
parser.add_argument("--n_hidden", type=int, default=256, help='num of hidden layer of BiLSTM')

parser.add_argument("--test_dir", type=str, default='images', help='path of images to predict')

args = parser.parse_args()
