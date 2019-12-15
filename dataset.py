import torch
from torch.utils import data
from torchvision import transforms
import numpy as np

import os
import lmdb
from io import BytesIO
from random import randint
from PIL import Image

import utils


class LmdbDataset(data.Dataset):
    '''
    Args:
        root: root dir
        voc_type: indicate the word case, should be in ['LOWER_CASE', 'UPPER_CASE', 'ALL_CASE']
        max_label_length: a pre-defined length, used for aligning variable-length sequences
        target_img_size: size of the resized img
    '''
    
    def __init__(self, root, voc_type, max_label_length, target_img_size, training=True):
        super().__init__()
        assert os.path.exists(root)
        self.env = lmdb.open(root, readonly=True, max_readers=16)
        assert self.env is not None, 'cannot create lmdb dataset from {}'.format(root)

        self.txn = self.env.begin()
        self.num_samples = int(self.txn.get(b'num-samples'))
        print('Totally {} samples in {}'.format(self.num_samples, root))
        self.voc_type = voc_type
        self.training = training
        self.char2id = utils.get_dict(dict_type='CHAR2ID', voc_type=self.voc_type)
        self.id2char = utils.get_dict(dict_type='ID2CHAR', voc_type=self.voc_type)
        self.num_classes = len(self.char2id)
        self.max_label_length = max_label_length
        if self.training:
            self.img_transforms = transforms.Compose([
                transforms.Resize(target_img_size),
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            self.img_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # keys in lmdb are created 1-indiced
        if index > len(self):
            index = randint(0, len(self) - 1)
        index += 1

        # restore img
        image_key = b'image-%09d' % index
        img_buf = self.txn.get(image_key)
        try:
            img = Image.open(BytesIO(img_buf)).convert('RGB')
        except IOError:
            print('Failed to load picture {}'.format(index))
            return self[index + 1]

        # restore label
        label_key = b'label-%09d' % index
        word = self.txn.get(label_key).decode()
        if self.voc_type == 'LOWER_CASE':
            word = word.lower()
        label_list = [self.char2id.get(char, self.char2id['UNKNOWN']) for char in word]
        label_list.append(self.char2id['EOS'])

        label_length = len(label_list)
        assert label_length < self.max_label_length
        # use fix-length self.max_label_length temporarily, more memory would be saved if max_label_len controlled via collate_fn
        label = np.full((self.max_label_length, ), self.char2id['PADDING'], dtype=np.int)
        label[:label_length] = np.array(label_list)

        if not self.training:
            w, h = img.size
            w = int(32 / h * w)
            img = img.resize((w, 32), Image.BILINEAR)

        return self.img_transforms(img), torch.tensor(label, dtype=torch.int, device='cpu'), torch.tensor(label_length, dtype=torch.int, device='cpu')


class MyCollateFn(object):
    
    def __init__(self, img_w, img_h):
        super().__init__()

    def __call__(self, batch):
        imgs, labels, label_lengths = zip(*batch)

        raise NotImplementedError


if __name__ == "__main__":
    root = 'D:/DeepLearning/datasets/synth90k/train/'
    # root = '/home/HuangCH/Synth90k/train'
    voc_type = 'LOWER_CASE'
    max_label_length = 64
    target_img_size = (32, 100)
    dataset = LmdbDataset(root, voc_type, max_label_length, target_img_size, training=False)

    dataloader = data.DataLoader(dataset, batch_size=1)

    for i, mini_batch in enumerate(dataloader):
        img, label, label_length = mini_batch
        print(type(img))
        print(type(label))
        print(type(label_length))
        print('img.shape = ', img.shape)
        print('label.shape = ', label.shape)
        print('label_length.shape = ', label_length.shape)
        print('label = ', label)
        print('label_length = ', label_length)
        input('==========================')