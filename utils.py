import string


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