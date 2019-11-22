import string


class Validator(object):
    def __init__(self, voc_type):
        self.voc_type = voc_type
        self.char2id = get_dict(dict_type='CHAR2ID', voc_type=self.voc_type)
        self.id2char = get_dict(dict_type='ID2CHAR', voc_type=self.voc_type)

        self.reset()
        
    def reset(self):
        self.precision = 0.
        self.correct_num = 0
        self.total_num = 0

    def update(self):
        self.precision = self.correct_num / self.total_num

    def validate(self, preds, labels, label_lengths):
        '''Validate the number of correct predictions
        Inputs:
            preds: network output followed by a argmax, (batch_size, max_label_length)
            labels: ground truth, (batch_size, max_label_length)
            label_lengths: ground truth. (batch_size, )
        '''
        preds, labels, label_lengths = preds.tolist(), labels.tolist(), label_lengths.tolist()
        for pred, label, label_length in zip(preds, labels, label_lengths):
            # print('pred = ', pred)
            # print('label = ', label)
            # print('label_lengths = ', label_length)
            if pred[:label_length] == label[:label_length]:
                self.correct_num += 1
        self.total_num += len(label_lengths)


class AverageMeter(object):
    '''compute and store the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


def get_dict(dict_type: str, voc_type: str,  EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN'):
    vocabulary = [EOS, PADDING, UNKNOWN]
    if voc_type.upper() == 'LOWER_CASE':
        vocabulary.extend(string.digits + string.ascii_lowercase)
    elif voc_type.upper() == 'UPPER_CASE':
        vocabulary.extend(string.digits + string.ascii_uppercase)
    elif voc_type.upper() == 'ALL_CASE':
        vocabulary.extend(string.digits + string.ascii_letters)
    else:
        raise KeyError('voc_type must be LOWER_CASE, UPPER_CASE or ALL_CASE')

    if dict_type.upper() == 'CHAR2ID':
        return {char: ID for ID, char in enumerate(vocabulary)}
    elif dict_type.upper() == 'ID2CHAR':
        return {ID: char for ID, char in enumerate(vocabulary)}


if __name__ == "__main__":
    voc_types = ['LOWER_CASE', 'UPPER_CASE', 'ALL_CASE']
    dict_types = ['CHAR2ID', 'ID2CHAR']
    for voc_type in voc_types:
        for dict_type in dict_types:
            dic = get_dict(dict_type, voc_type)
            print(len(dic))