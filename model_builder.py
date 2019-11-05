import torch
import time

from net import Seq2SeqNetwork
from utils import AverageMeter
from training_tools import TrainingTools
from dataset import LmdbDataset
from loss import SequenceLoss


class Seq2SeqTextRecognizer(object):
    '''
    The class for seq2seq-text-recognizer, including train, val, test and pred
    '''

    def __init__(self, args):
        self.args = args
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()

        self.eval_score = None

        self.net = None
        self.criterion = SequenceLoss()
        self.train_set = None
        self.train_loader = None
        self.val_set = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None

        self.device = None

        # store the info such as epoch, iter
        self.state = dict()

        self._init_model()
        
    def _init_model(self):
        self.device = torch.device('cuda' if self.args.cuda else 'cpu')

        # Prepare datasets and dataloaders
        self.train_set = LmdbDataset(root=self.args.train_root, voc_type=self.args.voc_type, max_label_length=self.args.max_label_length, target_img_size=(self.args.img_height, self.args.img_width))
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.args.batch_size, num_workers=self.args.num_workers)
        try:
            self.val_set = LmdbDataset(root=self.args.test_root, voc_type=self.args.voc_type, max_label_length=self.args.max_label_length, target_img_size=(self.args.img_height, self.args.img_width))
            self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.args.batch_size, num_workers=self.args.num_workers)
        except Exception as e:
            print(e)
            print('Start training without validation!')
            self.val_loader = None

        # Prepare the network
        self.net = Seq2SeqNetwork(self.args.hidden_size, self.train_set.num_classes, self.args.max_label_length)
        
        # Prepare optimizer and scheduler
        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError('Only Adam and SGD are supported now.')
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestones, gamma=0.1)

        # Resume or initiate training
        if self.args.resume:
            raise NotImplementedError
        else:
            self.state['epoch'] = 0
            self.state['iters'] = 0



    def train(self):
        ''' Train the model for an epoch '''
        self.net.train()
        start_time = time.time()
        self.state['epoch'] += 1

        for i, data_list in enumerate(self.train_loader):
            self.data_time.update(time.time() - start_time)
            data_list = TrainingTools.data_to_device(data_list, self.device)
            img, label, label_length = data_list

            # Forward
            out = self.net(img, label, label_length)
            loss = self.criterion(out, label, label_length)
            self.train_losses.update(loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the training info
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.state['iters'] += 1


            # Print the log info
            if self.state['iters'] % self.args.print_iter == 0:
                print('epoch: {}, iters: {}/{}, batch_time: {:.2f}, data_time: {:.3f}, avg_loss: {:.3f}'.format(
                    self.state['epoch'],
                    self.state['iters'],
                    len(self.train_loader),
                    self.batch_time.val,
                    self.data_time.val,
                    self.train_losses.avg
                ))
                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()
                input('====================================')

            # if self.state['iters'] % self.args.val_iter == 0:
            #     self.val()

    def val(self):
        ''' validate the model during training '''
        self.net.eval()
        start_time = time.time()
        eval_result = None
        with torch.no_grad():
            for i, data_dict in enumerate(self.val_loader):
                # Forward
                data_dict = TrainingTools.data_to_device(data_dict, self.device)
                out = self.net(data_dict)
                loss = self.criterion(out)
                self.val_losses.update(loss.item())

                # Calculate the precision
                raise NotImplementedError
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()
            
            print('Evaluation time: {}'.format(self.batch_time.sum))

            self.eval_score.reset()
            self.batch_time.reset()
            self.val_losses.reset()
            self.net.train()