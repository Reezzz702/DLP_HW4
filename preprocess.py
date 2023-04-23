import pandas as pd
from PIL import Image, ImageFile
import numpy as np
from tqdm import tqdm
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True


def preprocess(image):
    size = np.asarray(image.size)

    short = min(size)
    l = short//2
    center = size//2
    image = image.crop((int(center[0]) - l, int(center[1]) - l, int(center[0]) + l, int(center[1]) + l))
    image = image.resize((512, 512))
    return image

img = pd.read_csv('train_img.csv')
file_list = np.squeeze(img.values)

os.makedirs("train", exist_ok=True)
for file in tqdm(file_list):
    path = os.path.join("new_train", file + '.jpeg')
    img = preprocess(Image.open(path).convert('RGB'))
    save_path = os.path.join("train", file + '.jpeg')
    img.save(save_path)

img = pd.read_csv('test_img.csv')
file_list = np.squeeze(img.values)

os.makedirs("test", exist_ok=True)
for file in tqdm(file_list):
    path = os.path.join("new_test", file + '.jpeg')
    img = preprocess(Image.open(path).convert('RGB'))
    save_path = os.path.join("test", file + '.jpeg')
    img.save(save_path)