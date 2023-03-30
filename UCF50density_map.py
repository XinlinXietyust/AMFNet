import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

root = r'./dataset/UCF_CC_50'

#part_1_train = os.path.join(root, 'part_1/train_data', 'images')
#part_1_test = os.path.join(root, 'part_1/test_data', 'images')
#part_2_train = os.path.join(root, 'part_2/train_data', 'images')
#part_2_test = os.path.join(root, 'part_2/test_data', 'images')
#part_3_train = os.path.join(root, 'part_3/train_data', 'images')
#part_3_test = os.path.join(root, 'part_3/test_data', 'images')
#part_4_train = os.path.join(root, 'part_4/train_data', 'images')
#part_4_test = os.path.join(root, 'part_4/test_data', 'images')
part_5_train = os.path.join(root, 'part_5/train_data', 'images')
part_5_test = os.path.join(root, 'part_5/test_data', 'images')

#path_sets = [part_2_train, part_2_test,part_3_train, part_3_test,part_4_train, part_4_test,part_5_train, part_5_test]
path_sets = [part_5_train, part_5_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace(' ', ' '))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    #gt = mat["image_info"][0, 0][0, 0][0]
    gt = mat["annPoints"]
    count = 0
    for i in range(len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][1]) >= 0 and int(gt[i][0]) < img.shape[1] and int(gt[i][0]) >= 0:
            k[int(gt[i][1]), int(gt[i][0])] = 1
            count += 1
    print('Ignore {} wrong annotation.'.format(len(gt) - count))
    k = gaussian_filter(k, 5)
    att = k > 0.001
    att = att.astype(np.float32)
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'new_data'), 'w') as hf:
        hf['density'] = k
        hf['attention'] = att
        hf['gt'] = count