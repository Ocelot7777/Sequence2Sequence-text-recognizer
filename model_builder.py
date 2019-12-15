import torch
import os
import time

from tqdm import tqdm
from net import Seq2SeqNetwork
from utils import AverageMeter, Validator
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
        self.validator = Validator(self.args.voc_type)
        self.validator_naive = Validator(self.args.voc_type)

        self.net = None
        self.criterion = SequenceLoss()
        self.train_set = None
        self.train_loader = None
        self.val_set = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None

        self.device = None

        # store the info such as epoch, iter, best_precision
        self.state = dict()

        self._init_model()
        
    def _init_model(self):
        self.device = torch.device('cuda' if self.args.cuda else 'cpu')

        # Prepare datasets and dataloaders
        self.train_set = LmdbDataset(root=self.args.train_root, voc_type=self.args.voc_type, max_label_length=self.args.max_label_length, target_img_size=(self.args.img_height, self.args.img_width), training=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        try:
            self.val_set = LmdbDataset(root=self.args.test_root, voc_type=self.args.voc_type, max_label_length=self.args.max_label_length, target_img_size=(self.args.img_height, self.args.img_width), training=False)
            self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=1, num_workers=self.args.num_workers)
        except Exception as e:
            print(e)
            print('Start training without validation!')
            self.val_loader = None

        # Prepare the network
        self.net = Seq2SeqNetwork(self.args.hidden_size, self.train_set.num_classes, self.args.max_label_length, self.args.with_attention, self.args.backbone).to(self.device)
        
        # Prepare optimizer and scheduler
        print('Selected optimizer is {}.'.format(self.args.optimizer))
        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(self.net.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError('Only Adam and SGD are supported now.')
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestones, gamma=0.1)

        # Resume or initiate training
        if self.args.resume:
            self.net.load_state_dict(torch.load(self.args.resume))
        # else:
        self.state['epoch'] = 0
        self.state['iters'] = 0
        self.state['best_precision'] = 0.



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


            # Save the model
            if self.state['iters'] % self.args.save_iter == 0:
                save_file_name = 'model_iters_{}_epoch_{}_loss_{:.3f}.pth'.format(self.state['iters'], self.state['epoch'], self.train_losses.avg)
                save_path = os.path.join(self.args.output_path, save_file_name)
                torch.save(self.net.state_dict(), save_path)
                print('Model has been saved to {}'.format(save_path))

            # Print the log info
            if self.state['iters'] % self.args.print_iter == 0:
                print('epoch: {}, iters: {}/{}, batch_time: {:.2f}, data_time: {:.3f}, avg_loss: {:.3f}, lr: {}, best_precision: {:.4f}'.format(
                    self.state['epoch'],
                    self.state['iters'],
                    len(self.train_loader),
                    self.batch_time.val,
                    self.data_time.val,
                    self.train_losses.avg,
                    self.optimizer.param_groups[0]['lr'],
                    self.state['best_precision']
                ))
                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            # Validation
            if self.state['iters'] % self.args.val_iter == 0:
                self.validate()

    def validate(self):
        original_state = self.net.teacher_forcing

        print('Start validation without teacher forcing.')
        save_flag = True
        self.net.teacher_forcing = False
        self.val_epoch(save_flag)

        print('Start validation with teacher forcing.')
        save_flag = False
        self.net.teacher_forcing = True
        self.val_epoch(save_flag)

        self.net.teacher_forcing = original_state

    def val_epoch(self, save_flag):
        ''' validate the model during training '''
        self.net.eval()
        start_time = time.time()
        with torch.no_grad():
            for data_list in tqdm(self.val_loader):
                data_list = TrainingTools.data_to_device(data_list, self.device)
                img, label, label_length = data_list

                # Forward
                out = self.net(img, label, label_length)
                loss = self.criterion(out, label, label_length)
                self.val_losses.update(loss.item())

                # Validate the prediction
                self.validator.validate(torch.argmax(out, dim=-1), label, label_length)
                self.batch_time.update(time.time() - start_time)
                start_time = time.time()

            self.validator.update()
            
            # Print the log info
            print('Evaluation time: {}'.format(self.batch_time.sum))
            print('Correct/Total: {}/{}, Precision: {:.4f}'.format(self.validator.correct_num, self.validator.total_num, self.validator.precision))
            print('Validation loss: {:.3f}'.format(self.val_losses.avg))

            # Save the model
            if save_flag and self.validator.precision > self.state['best_precision']:
                self.state['best_precision'] = self.validator.precision
                save_file_name = 'model_best_precision_{:.4f}_loss_{:.3f}.pth'.format(self.validator.precision, self.val_losses.avg)
                save_file_path = os.path.join(self.args.output_path, save_file_name)
                torch.save(self.net.state_dict(), save_file_path)
                print('Best model has been saved to {}'.format(save_file_path))

            self.validator.reset()
            self.batch_time.reset()
            self.val_losses.reset()
            self.net.train()