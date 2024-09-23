#split the shot file and query file

import numpy as np
import torch
from utils import ensure_path
import os, glob
import random, csv
from PIL import Image
from torch.utils.data import Dataset


filename = 'dataset/GID5_new'
root = '.'
filename_shot = './dataset/GID5_new/shot_GID5'   # shot folder name
filename_query = './dataset/GID5_new/query_GID5'  # query folder name
shot = 5

class SOURCE_DATA(Dataset):

    def __init__(self, setname):
        super(SOURCE_DATA, self).__init__()
        self.root = os.path.join(root, setname)
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(self.root))):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue
            if name == 'labels':
                self.name2label[name] = len(self.name2label.keys())
        self.data, self.label = self.load_csv('source.csv')
        print('load data:')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 'pokemon\\mewtwo\\00001.png
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                images += glob.glob(os.path.join(self.root, name, '*.tif'))
                images += glob.glob(os.path.join(self.root, name, '*.png'))

            # 1167, 'pokemon\\bulbasaur\\00000000.png'
            print(len(images), images)

            #random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels

#
def Split_shot_query(root,filename,file_shot,file_query,shot, num_split = 20, num_way = 5):

    filename = filename
    number_count = 0
    for loop in range(0,200):
        filename_shot = os.path.join(file_shot, str(shot)+'_shot'+ str(number_count) + '.csv')
        filename_query = os.path.join(file_query, str(shot)+'_query'+ str(number_count) + '.csv')
        if not os.path.exists(os.path.join(root, filename)):
            print('csv file dose not exist!')
        else:
            images = []
            with open(os.path.join(root, filename)) as f:
                reader = csv.reader(f)
                for row in reader:
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    img, label = row

                    images.append(img)

                shot_stack = []
                query_stack = []
                l = len(images)
                pos_temp = torch.randperm(l)   #pos = torch.randperm(len(l))[:self.n_per]
                pos1 = pos_temp[:shot]
                pos2 = pos_temp[shot:]

                image_temp = []
                for i in range(0,len(pos1)):
                    image_temp_temp = Image.open(images[pos1[i]]).convert('RGB')
                    image_temp_temp = np.asarray(image_temp_temp, np.float32)
                    image_temp.append(image_temp_temp)

                image_temp = np.stack(image_temp)
                image_temp = torch.from_numpy(image_temp)
                judge_temp = torch.unique(image_temp)

                if len(judge_temp) == num_way:
                    for i_shot in range(0, len(pos1)):
                        shot_stack.append(pos1[i_shot])
                    for i_query in range(0, len(pos2)):#batch.append(l[pos])
                        query_stack.append(pos2[i_query])       # batch = torch.stack(batch).t().reshape(-1)
                    shot_stack = torch.stack(shot_stack).t().reshape(-1)
                    query_stack = torch.stack(query_stack).t().reshape(-1)

        if len(shot_stack):
            with open(os.path.join(root, filename_shot), mode='w', newline='') as f:
                writer = csv.writer(f)
                temp = shot_stack.detach().numpy()
                temp = temp.astype(np.int32)
                for i in range(0,len(temp)):
                    image_name = images[temp[i]].split('/')
                    image_name = image_name[4]
                    image_name = image_name[:-4]
                    print(image_name)
                    writer.writerow([image_name])
                print('writen shot csv file:', filename)
            with open(os.path.join(root, filename_query), mode='w', newline='') as f:
                writer = csv.writer(f)
                temp = query_stack.detach().numpy()
                temp = temp.astype(np.int32)
                for i in range(0,len(temp)):
                    image_name = images[temp[i]].split('/')
                    image_name = image_name[4]
                    image_name = image_name[:-4]
                    writer.writerow([image_name])
                print('writen query csv file:', filename)
            number_count = number_count + 1
        if number_count > num_split:
            break

if __name__=='__main__':
    SOURCE_DATA(os.path.join(root, filename))
    ensure_path(filename_shot)
    ensure_path(filename_query)
    Split_shot_query(root, os.path.join(filename,'source.csv'),filename_shot,filename_query,shot)
    pass
