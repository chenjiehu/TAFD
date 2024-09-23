import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import transform
import torch
import torchvision
import cv2 as cv
from torch.utils import data
from PIL import Image
import numpy as np


value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]
# Train
scale_min = 0.5
scale_max = 2.0
rotate_min = -10
rotate_max = 10
train_h = 256
train_w = 256
train_transform = transform.Compose([
    transform.RandScale([scale_min, scale_max]),
    transform.RandRotate([rotate_min, rotate_max], padding=mean, ignore_label=255),
    transform.RandomGaussianBlur(),
    transform.RandomHorizontalFlip(),
    transform.Crop([train_h, train_w], crop_type='rand', padding=mean, ignore_label=255),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)])

test_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)])

class DataSet_GID5(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(256, 256), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s.tif" % name)
            label_file = osp.join(self.root, "labels/%s.tif" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        size = image.shape
        image, label = test_transform(image, label)

        return image, label, np.array(size), name

class DataSet_GID5_tr(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(256, 256), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.transform = train_transform

        # self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        #                       19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
        #                       26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s.tif" % name)
            label_file = osp.join(self.root, "labels/%s.tif" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        size = image.shape
        image, label = train_transform(image, label)
        # print(np.unique(label.copy()))
        # return image.copy(), label_copy.copy(), np.array(size), name # 08.31 -1
        return image, label, np.array(size), name

class DataSet_GID5_te(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(256, 256), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s.tif" % name)
            label_file = osp.join(self.root, "labels/%s.tif" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        image, label = test_transform(image, label)
        size = image.shape
        # image = image[:, :, ::-1]  # change to BGR
        # image -= self.mean
        # image = image/((np.max(image)-np.min(image))+1.0e-12)
        # image = image.transpose((2, 0, 1))

        return image, label, np.array(size), name

def get_shot_data(root, shot_list_path, max_iters=None, crop_size=(256, 256), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255):

        list_path = shot_list_path
        crop_size = crop_size
        scale = scale
        ignore_label = ignore_label
        mean = mean
        is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            img_ids = img_ids * int(np.ceil(float(max_iters) / len(img_ids)))
        files = []

        for name in img_ids:
            img_file = osp.join(root, "images/%s.tif" % name)
            label_file = osp.join(root, "labels/%s.tif" % name)
            files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        image_stack = []
        label_stack = []
        name_stack = []

        for index in range(0,len(img_ids)):

            datafiles = files[index]

            image = Image.open(datafiles["img"]).convert('RGB')
            label = Image.open(datafiles["label"])
            name = datafiles["name"]

            # resize
            image = image.resize(crop_size, Image.BICUBIC)
            label = label.resize(crop_size, Image.NEAREST)

            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)

            size = image.shape
            image, label = test_transform(image, label)

            image_stack.append(image)
            label_stack.append(label)
            name_stack.append(name)

        image_stack = torch.from_numpy(np.stack(image_stack))
        label_stack = torch.from_numpy(np.stack(label_stack))
        name_stack = np.stack(name_stack)

        return image_stack, label_stack, np.array(size), name_stack

def get_shot_data_tr(root, shot_list_path, max_iters=None, crop_size=(256, 256), mean=(128, 128, 128), scale=True,
                  mirror=True, ignore_label=255):
    list_path = shot_list_path
    crop_size = crop_size
    scale = scale
    ignore_label = ignore_label
    mean = mean
    is_mirror = mirror
    # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    img_ids = [i_id.strip() for i_id in open(list_path)]
    if not max_iters == None:
        img_ids = img_ids * int(np.ceil(float(max_iters) / len(img_ids)))
    files = []

    for name in img_ids:
        img_file = osp.join(root, "images/%s.tif" % name)
        label_file = osp.join(root, "labels/%s.tif" % name)
        files.append({
            "img": img_file,
            "label": label_file,
            "name": name
        })
    image_stack = []
    label_stack = []
    name_stack = []

    for index in range(0, len(img_ids)):
        datafiles = files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(crop_size, Image.BICUBIC)
        label = label.resize(crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        image, label = train_transform(image, label)

        size = image.shape
        # image = image[:, :, ::-1]  # change to BGR
        # image = image.transpose((2, 0, 1))

        image_stack.append(image)
        label_stack.append(label)
        name_stack.append(name)

    image_stack = torch.from_numpy(np.stack(image_stack))
    label_stack = torch.from_numpy(np.stack(label_stack))
    name_stack = np.stack(name_stack)

    return image_stack, label_stack, np.array(size), name_stack

def get_shot_data_te(root, shot_list_path, max_iters=None, crop_size=(256, 256), mean=(128, 128, 128), scale=True,
                  mirror=True, ignore_label=255):
    list_path = shot_list_path
    crop_size = crop_size
    scale = scale
    ignore_label = ignore_label
    mean = mean
    is_mirror = mirror
    # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    img_ids = [i_id.strip() for i_id in open(list_path)]
    if not max_iters == None:
        img_ids = img_ids * int(np.ceil(float(max_iters) / len(img_ids)))
    files = []

    for name in img_ids:
        img_file = osp.join(root, "images/%s.tif" % name)
        label_file = osp.join(root, "labels/%s.tif" % name)
        files.append({
            "img": img_file,
            "label": label_file,
            "name": name
        })
    image_stack = []
    label_stack = []
    name_stack = []

    for index in range(0, len(img_ids)):
        datafiles = files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(crop_size, Image.BICUBIC)
        label = label.resize(crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        image, label = test_transform(image, label)

        size = image.shape
        # image = image[:, :, ::-1]  # change to BGR
        # image = image.transpose((2, 0, 1))

        image_stack.append(image)
        label_stack.append(label)
        name_stack.append(name)

    image_stack = torch.from_numpy(np.stack(image_stack))
    label_stack = torch.from_numpy(np.stack(label_stack))
    name_stack = np.stack(name_stack)

    return image_stack, label_stack, np.array(size), name_stack


color2index_GID5 = {
    (0, 0, 255):0,    # building
    (0, 255, 0):1,   # farmland
    (255, 255, 0):2, # forest
    (255, 0, 0):3, # water
    (0, 0, 0):4, # other
}


index2color_GID5 = {
    0: (0, 0, 255),  # building
    1: (0, 255, 0),  # farmland
    3: (255, 255, 0),  # forest
    2: (255, 0, 0),  # water
    4: (0, 0, 0),  # other
}


def write_fname_txt(path, list_path):   # write names into a txt file
    filepath = path   #obtain the images' path
    name_list = os.listdir(filepath)   #scan files in directory and return a list
    file_name = []   #define an empty list
    for i in name_list:
        file_name.append(i.split('.')[0])
    file_name.sort()

    for i in file_name:
        with open(os.path.join(list_path,'label_list.txt'),'a') as f:
            f.write(i+'\n')
        f.close()
    print(file_name)



if __name__ == '__main__':
    path = './dataset/GID5/labels'    #source path (contain images)
    path_list = './dataset/GID5'

    root = './dataset/GID5/labels_RGB'
    path_index = './dataset/GID5/labels'
    paths = os.listdir(root)
    count = 0
    for path in paths:
        a, b = os.path.splitext(path)
        this_dir = os.path.join(root, path)  # 构建保存 路径+文件名

        img = cv.imread(this_dir)

    #
        img_index = im2index_GID5(img, class_number=5)
        out_path = os.path.join(path_index, path)
        cv.imwrite(out_path, img_index)





