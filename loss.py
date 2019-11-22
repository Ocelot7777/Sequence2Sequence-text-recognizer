import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceLoss(nn.Module):
    '''
    Args:
        None
    Input:
        pred: predicted label sequence, (batch_size, max(label_lengths), output_size=num_classes)
        labels: sequence labels, (batch_size, config.pre_defined_length)
        lengths: length for each sequence, (batch_size, )
    '''
    def __init__(self):
        super().__init__()
        self.loss = nn.NLLLoss(reduction='none')

    def forward(self, pred, label, lengths):
        batch_size , max_length = label.size(0), max(lengths)
        # only the valid part of sequence should be calculated into loss
        mask = torch.zeros((batch_size, max_length))
        for i in range(batch_size):
            mask[i, :lengths[i]].fill_(1)
        mask = mask.type_as(pred)

        # assert pred.size(1) == max_length
        label = label[:, :max_length].long()

        # (N, max_seq_len, C) --> (N, C, max_seq_len)
        pred = pred.transpose(1, 2)

        output = self.loss(pred, label) * mask
        
        eps = 1e-5
        return torch.sum(output) / (torch.sum(mask) + eps)


if __name__ == "__main__":
    batch_size = 5
    max_length = 12
    num_classes = 38
    pred = torch.randn((batch_size, max_length, num_classes))
    label = torch.randint(low=0, high=num_classes, size=(batch_size, max_length * 2))
    lengths = torch.randint(low=1, high=max_length, size=(batch_size, ))
    lengths[-2] = max_length
    print('lengths = ', lengths)
    print(lengths[-1])
    print('pred.shape = ', pred.shape)
    print('label.shape = ', label.shape)
    criterion = SequenceLoss()
    loss = criterion(pred, label, lengths)
    print('loss.shape = ', loss.shape)