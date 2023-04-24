import pandas as pd
from PIL import Image, ImageFile
import numpy as np
from tqdm import tqdm
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True


def preprocess(image):
    # Calculate the shorter side of the original rectangle
    width, height = image.size
    short_side = min(width, height)

    # Crop the image to a square shape
    left = (width - short_side) // 2
    top = (height - short_side) // 2
    right = (width + short_side) // 2
    bottom = (height + short_side) // 2
    image = image.crop((left, top, right, bottom))

    # Resize the square image to the desired size
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