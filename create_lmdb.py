import os
import lmdb
import cv2
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import time
from scipy import io as sio


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    try:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print('image open, Exception: ', e)
        return False
    try:
        imgH, imgW = img.shape[0], img.shape[1]
    except Exception as e:
        print(e)
        return False
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def create_dataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in tqdm(range(nSamples)):
        imagePath = imagePathList[i]
        label = labelList[i]
        if len(label) == 0:
            continue
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def get_list(txt_name, root_dir):
    txt_path = os.path.join(root_dir, txt_name)
    if not os.path.exists(txt_path):
        raise FileNotFoundError('%s does not exist' % txt_path)
    img_paths = []
    labels = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    count = 0
    for line in tqdm(lines):
        img_path = line.split()[0].replace('./', '')
        label = img_path.split('_')[1]
        img_paths.append(os.path.join(root_dir, img_path))
        labels.append(label)
        count += 1
    return img_paths, labels

def synth90k():
    root_dir = '/home/HuangCH/Synth90k/'
    annotation_path = os.path.join(root_dir, '90kDICT32px')
    train_txt_name = 'annotation_train.txt'
    val_txt_name = 'annotation_val.txt'
    
    train_output_path = os.path.join(root_dir, 'train')
    val_output_path = os.path.join(root_dir, 'val')

    print('start getting list')
    train_img_paths, train_labels = get_list(train_txt_name, annotation_path)
    val_img_paths, val_labels = get_list(val_txt_name, annotation_path)
    print('finish getting list')

    print('start creating dataset')
    create_dataset(train_output_path, train_img_paths, train_labels, checkValid=True)
    create_dataset(val_output_path, val_img_paths, val_labels, checkValid=True)
    print('finish creating dataset')

def IIIT5k():
    train_output_path = './lmdb_train/'
    test_output_path = './lmdb_test/'

    print('start loading mat')
    train_data_mat_path = './traindata.mat'
    test_data_mat_path = './testdata.mat'
    train_data_mat = sio.loadmat(train_data_mat_path)
    test_data_mat = sio.loadmat(test_data_mat_path)
    print('finished loading mat')

    train_img_paths = []
    train_labels = []
    test_img_paths = []
    test_labels = []

    for train_sample in tqdm(train_data_mat['traindata'][0]):
        image_path = train_sample[0][0]
        label = train_sample[1][0]

        train_img_paths.append(image_path)
        train_labels.append(label)

    for test_sample in tqdm(test_data_mat['testdata'][0]):
        image_path = test_sample[0][0]
        label = test_sample[1][0]

        test_img_paths.append(image_path)
        test_labels.append(label)

    # print(test_img_paths[:20])
    # print(test_labels[:20])

    print('start creating lmdb')
    create_dataset(train_output_path, train_img_paths, train_labels)
    create_dataset(test_output_path, test_img_paths, test_labels)
    print('finished creating lmdb')




if __name__ == "__main__":
    # start_time = time.time()
    # synth90k()
    # end_time = time.time()
    # print('time cost = {:.3f}'.format(end_time - start_time))
    IIIT5k()