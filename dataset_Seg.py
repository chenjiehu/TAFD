import pandas as pd
import os
import torch as t
import numpy as np
from skimage.segmentation import slic,mark_boundaries

def image_seg(img, label, seg_number):
    img_int = img.permute(1,2,0)
    img_int = img_int.cpu().numpy()

    segments = slic(img_int, n_segments=seg_number, compactness=20, enforce_connectivity=True, convert2lab=True)


    Number_seg = np.unique(segments)
    max_label = np.max(segments)
    segments_refine = segments.copy()
    for j in range(len(Number_seg)):
        position_j = np.where(segments == Number_seg[j])
        position_j = list(position_j)
        label_region = label[position_j]
        # label_region = label_region[:,0]
        label_temp = label_region.tolist()
        if not len(label_temp):
            continue
        mask_set = set(label_temp)
        mask_set_len = len(mask_set)
        mask_set = list(mask_set)
        if mask_set_len > 1:
            for ii in range(mask_set_len):
                sub_mask_position = np.where(label_region.cpu().numpy() == mask_set[ii])
                sub_mask_position = list(sub_mask_position)
                max_label = max_label + 1
                sub_mask_len = len(sub_mask_position[0])
                temp = []
                position_x = []
                position_y = []
                for jj in range(sub_mask_len):
                    position_temp = sub_mask_position[0][jj]
                    position_x.append(position_j[0][position_temp])
                    position_y.append(position_j[1][position_temp])
                position_jj = [np.array(position_x),np.array(position_y)]
                segments_refine[position_x,position_y] = max_label
    return segments, segments_refine

def images_seg(imgs_concat, mask_concat, seg_number = 60):

    # imgs_concat = torch.cat([supp_imgs, qry_imgs], dim=0)
    # mask_concat = torch.cat([supp_mask, qry_maks], dim=0)

    img_segments = []
    img_segments_refine = []
    for i in range(imgs_concat.shape[0]):
        img_segments_temp = image_seg(imgs_concat[i], mask_concat[i], seg_number=seg_number)
        img_segments.append(img_segments_temp[0])
        img_segments_refine.append(img_segments_temp[1])
        test_nuique1 = np.unique(img_segments_temp[0])
        test_unique2 = np.unique(img_segments_temp[1])
    # img_segments = torch.cat(img_segments,dim=0)
    img_segments = np.array(img_segments)
    img_segments_refine = np.array(img_segments_refine)
    return img_segments, img_segments_refine


class LabelProcessor:

    def __init__(self, file_path):

        self.colormap = self.read_color_map(file_path)

        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap):
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = 1
        return cm2lbl

    def encode_label_img(self, img):

        data = np.array(img, dtype='int32')
        idx = (data[:,:,0]*256 + data[:,:,1])*256 + data[:,:,2]
        return np.array(self.cm2lbl[idx], dtype='int64')


# class LoadDataset(Dataset):
#     def __init__(self, file_path=[], crop_size=None):
#
#         if len(file_path) != 2:
#             raise ValueError("同时需要图片和标签的路径，图片路径在前")
#         self.img_path = file_path[0]
#         self.label_path = file_path[1]
#
#         self.imgs = self.read_file(self.img_path)
#         self.labels = self.read_file(self.label_path)
#         self.crop_size = crop_size
#
#     def __getitem__(self, index):
#         img = self.imgs[index]
#         label = self.labels[index]
#
#         img = Image.open(img)
#         label = Image.open(label).convert('RGB')
#
# #        img, label = self.center_crop(img, label, self.crop_size)
#
#         img, label = self.img_transform(img, label)
#
#         sample = {'img':img, 'label': label}
#
#         return sample
#
#     def __len__(self):
#         return len(self.imgs)
#
#     def read_file(self, path):
#          files_list = os.listdir(path)
#          file_path_list = [os.path.join(path, img) for img in files_list]
#          file_path_list.sort()
#          return file_path_list
#
#     def center_crop(self, data, label, crop_size):
#          data = ff.center_crop(data, crop_size)
#          label = ff.center_crop(label, crop_size)
#          return data, label
#
#     def img_transform(self, img, label):
#
#         label = np.array(label)
#         label = Image.fromarray(label.astype('uint8'))
#         transfrom_img = transfroms.Compose(
#             [
#                 transfroms.ToTensor(),
#                 transfroms.Normalize([0.485, 0.485, 0.485],[0.229, 0.224, 0.225])
#             ]
#         )
#         img = transfrom_img(img)
#         label = label_processor.encode_label_img(label)
#         label = t.from_numpy(label)
#
#         return img, label
#     def denormalize(self, x_hat):
#
#         mean = [0.485, 0.485, 0.485]
#         std = [0.229, 0.224, 0.225]
#
#         mean = t.tensor(mean).unsqueeze(1).unsqueeze(1)
#         std = t.tensor(std).unsqueeze(1).unsqueeze(1)
#         x = x_hat * std + mean
#
#         return x

