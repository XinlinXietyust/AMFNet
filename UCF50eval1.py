#import matplotlib.pyplot as plt 
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import imageio
import numpy as np
from PIL import Image as pil_image
from torchvision import transforms
#img = torch.randn(3,128,64)
import cv2

from skimage import io
import torch
from torch.utils import data
from UCF50dataset import Dataset
from modelsSFANet import Model
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SHB', type=str, help='dataset')
parser.add_argument('--data_path', default=r'./dataset', type=str, help='path to dataset')
parser.add_argument('--save_path', default=r'./checkpoint/SFANet', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')

args = parser.parse_args()

test_dataset = Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:' + str(args.gpu))

model = Model().to(device)

checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))
model.load_state_dict(checkpoint['model'])

model.eval()
toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
with torch.no_grad():
    mae, mse = 0.0, 0.0
    for i, (images, gt) in enumerate(test_loader):
        images = images.to(device)

        predict, _ = model(images)

        pred = torch.sum(predict).item()
        predict = predict.cpu().data.numpy()[0,0,:,:]
        #pred = torch.sum(predict).item()
        algt = torch.sum(gt).item()
        filename='./1'
        cmap = plt.cm.jet
        #cmap = plt.get_cmap('jet')
        #cmap=plt.cm.get_cmap('jet')
        #plt.imsave(os.path.join('filename', predict, cmap=plt.get_cmap('jet')))
        plt.imsave(os.path.join('filename','{}.png'.format(i)), predict,cmap='jet')
        #plt.imsave(os.path.join('pred', f'[{filename}]_[{pred:.2f}|{algt:.2f}]_[{psnr:.2f}]_[{ssim:.4f}].png'), predict, cmap=cmap)
        #plt.imsave(os.path.join('predict', f'[{filename}]_[{predict:.2f}|{algt:.2f}]_[{psnr}]_[{ssim:.4f}].png'), predict, cmap='jet')
        #predict = predict.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        #imageio.imsave('./distance_map.pgm', predict)
        #predict1 = predict .squeeze(0)
        #print(predict1.shape)
        #norm_img = np.zeros(predict.shape)
        #norm_img = cv2.normalize(predict , norm_img, 0, 255, cv2.NORM_MINMAX)
        #norm_img = np.asarray(norm_img, dtype=np.uint8)
        #heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
        #heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)# 将BGR图像转为RGB图像
        #heat_img.save('./44.jpg')

        #toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
        #pic = toPIL(predict1)
        #pic.save('./43.jpg')
        #predicts = predict.permute(0,2,3,1)
        #predictss = predicts.squeeze(0)
        #k = predictss.cpu().detach().numpy()
        #print(k)
        #image = pil_image.fromarray(np.uint8(k)).convert('RGB')
        #image = pil_image.fromarray(np.uint8(k))
			#image.show()
        #timestamp = datetime.datetime.now().strftime("%M-%S")
        #savepath = timestamp + '_r.jpg'
        #image.save(savepath)
        #predictss = predicts.squeeze(0)
        #pic = toPIL(predictss)
        #pic.save('./41.jpg')
        #print(predict)
        #print(predict.shape)
        #print(type(predict))
        #print(predict.dtype)
        #print('predict:{:.2f} label:{:.2f}'.format(predict.sum().item(), gt.item()))
        #mae += torch.abs(predict.sum().to(device) - gt.to(device)).item()
        #mse += ((predict.sum().to(device) - gt.to(device)) ** 2).item()

    #mae /= len(test_loader)
    #mse /= len(test_loader)
    #mse = mse ** 0.5
    #print('MAE:', mae, 'MSE:', mse)