import glob
import os
import os.path as osp
from tqdm import tqdm
import shutil
from sklearn.model_selection import ShuffleSplit
import json
from PIL import Image
import numpy as np
from loguru import logger


#
#图像数据集的均值: [234.30045203 225.71031344 235.95306603]
#图像数据集的方差: [ 8.5827689  35.25244448 38.0740568 ]
#
root = '/home/wjx/data/dataset/STAMP/raw'
imgs = glob.glob(root+'/*/*/*.jpg')
logger.add('/home/wjx/data/dataset/STAMP/raw/meta.log')
save = '/home/wjx/data/dataset/STAMP/processed/images'
mm = '/home/wjx/data/dataset/STAMP/processed/meta'
allnames = []
image_data = []
# 初始化变量来累积像素数据的总和与平方和
total_pixels = 0
mean = np.zeros(3)  # 假设是RGB图像，3通道
std = np.zeros(3)
total_width = 0
total_height = 0
image_count = 0


for img in tqdm(imgs):
    name_snipt = img.split('/')
    newname = name_snipt[-3] +'#'+name_snipt[-2]+'#'+name_snipt[-1]
    categroy = name_snipt[-3]
    train_val_index = categroy+'/'+newname
    newnamedir = osp.join(save,categroy)
    os.makedirs(newnamedir,exist_ok=True)
    #shutil.copy(img,osp.join(newnamedir,newname))
    allnames.append(train_val_index)
    with Image.open(img) as npimg:
        img_array = np.array(npimg)
        # 获取图像的宽度和高度
        width, height = npimg.size
        total_width += img_array.shape[1]
        total_height += img_array.shape[0]
        for c in range(3):
            mean[c] += img_array[:, :, c].mean()
            std[c] += img_array[:, :, c].std()
        

# 计算平均宽度和高度
average_width = total_width / len(imgs)
average_height = total_height / len(imgs)
mean_value = mean/len(imgs)
variance_value = std/len(imgs)
logger.info(f"图像数据集的均值: {mean_value}")
logger.info(f"图像数据集的方差: {variance_value}")
logger.info(f"图像数据集的平均宽度: {average_width}")
logger.info(f"图像数据集的平均高度: {average_height}")


rs = ShuffleSplit(n_splits=3)
tran_val = {}
for index,(train,test) in enumerate(rs.split(allnames)):
    train_index = [allnames[int(i)] for i in train]
    test_index = [allnames[int(i)] for i in test]
    tran_val['Fold_%s'%index] = {}
    tran_val['Fold_%s'%index]['train'] = train_index
    tran_val['Fold_%s'%index]['test'] = test_index
with open("/home/wjx/data/dataset/STAMP/processed/meta/meta.json", 'w', encoding='utf-8') as fw: 
 json.dump(tran_val, fw)
