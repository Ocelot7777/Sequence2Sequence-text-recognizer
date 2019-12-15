import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from net import Seq2SeqNetwork
from utils import get_dict


def recognize_one_image(model, img, label, label_length, id2char):
    with torch.no_grad():
        preds = model(img, label, label_length)
        print('preds.shape = {}'.format(preds.shape))
        # assert preds.dim() == 2
        preds = torch.argmax(preds, dim=-1).tolist()
        # result_str = ''.join([id2char[c] for c in preds[0]])
        result_str = [id2char[c] for c in preds[0]]
        print('Original preds = {}'.format(preds))
        print('Result string is = {}'.format(result_str))

        

if __name__ == "__main__":
    ID2CHAR = get_dict('ID2CHAR', 'ALL_CASE')
    img_path = r'D:\DeepLearning\datasets\IIIT5K\test/328_21.png'
    pth_path = r'D:\download/model_best_precision_0.3523_loss_4.027.pth'
    hidden_size = 256
    output_size = len(ID2CHAR)
    max_label_length = 64

    model = Seq2SeqNetwork(hidden_size, output_size, max_label_length, with_attention=False, backbone='temp_net', teacher_forcing=False)
    model.load_state_dict(torch.load(pth_path))
    img_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img = Image.open(img_path)
    w, h = img.size
    w = int(32 / h * w)
    img = img.resize((w, 32), Image.BILINEAR)
    img = img_transforms(img).unsqueeze(0)
    fake_label = [0]
    fake_label_length = [32]

    recognize_one_image(model, img, fake_label, fake_label_length, ID2CHAR)