import torch
import argparse
import os

from model_builder import Seq2SeqTextRecognizer


def train(args):
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    model = Seq2SeqTextRecognizer(args)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # Prepare datasets and dataloaders

    # Prepare optimizer and scheduler

    # Start training loop
    if args.resume:
        model.validate()
    for epoch in range(args.max_epoch):
        model.train()
        model.scheduler.step()

        if (epoch + 1) % args.save_epoch == 0:
            save_file_path = os.path.join(args.output_path, 'epoch_{}.pth'.format(epoch))
            torch.save(model.net.state_dict(), save_file_path)


def test(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameters.')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--cuda', type=bool, default=True)

    # Parameters for dataset
    parser.add_argument('--train_root', type=str, help='Root directory of the training_set.')
    parser.add_argument('--test_root', type=str, help='Root directory of the test_set.')
    parser.add_argument('--voc_type', type=str, default='ALL', choices=['LOWER_CASE', 'UPPER_CASE', 'ALL_CASE', 'ALL'])
    parser.add_argument('--img_height', type=int, default=32, help='Target height of the resized input image.')
    parser.add_argument('--img_width', type=int, default=128, help='Target width of the resized input image.')

    # Parameters for dataLoader
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)

    # Parameters for optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'Adadelta'])
    parser.add_argument('--lr', type=float, default=1.)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--milestones', type=list, default=[2, 4], help='Milestones used for MultiStepLR scheduler.')

    # Parameters for network
    parser.add_argument('--resume', type=str, default='', help='pth path.')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone used in encoder to extract features.')
    # parser.add_argument('--with_attention', type=bool, default=True, help='Whether to use attention.')
    parser.add_argument('--with_attention', action='store_true', help='Whether to use attention.')
    parser.add_argument('--hidden_size', type=int, default=256, help='The hidden_size used for encoder and decoder.')
    parser.add_argument('--output_size', type=int, default=38, help='The output_size used for seq2seq network, usually equals to num_classes + 2.')
    parser.add_argument('--max_label_length', type=int, default=64, help='A pre-defined length, used for aligning variable-length sequences.')

    # Parameters for training
    parser.add_argument('--print_iter', type=int, default=10)
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--save_iter', type=int, default=10000)
    parser.add_argument('--val_iter', type=int, default=10000)
    parser.add_argument('--output_path', type=str, default='./outputs/')

    args = parser.parse_args()

    if args.mode.lower() in ['train', 'training']:
        train(args)
    else:
        raise NotImplementedError('Only training mode is implemented now!')