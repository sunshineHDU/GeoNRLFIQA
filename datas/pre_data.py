import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from config import config
count_win5 = 140
count_nbu = 134
count_mpi = 214


class IQA_datset(torch.utils.data.Dataset):
    def __init__(self, config, scene_list, transform, mode='train'):
        super(IQA_datset, self).__init__()
        self.config = config
        self.scene_list = scene_list
        self.transform = transform
        self.mode = mode
        self.dis_path = self.config.db_path
        self.txt_file_name = self.config.text_path
        self.aug_num = self.config.aug_num
        idx_data, dis_files_data, score_data = [], [], []
        if config.db_name == 'win5':
            name_list_heng = [['5_1', '5_2', '5_3'], ['5_4', '5_5', '5_6'], ['5_7', '5_8', '5_9']]   #0
            name_list_shu = [['1_5', '2_5', '3_5'], ['4_5', '5_5', '6_5'], ['7_5', '8_5', '9_5']]   # 90
            name_list_pie = [['1_1', '2_2', '3_3'], ['4_4', '5_5', '6_6'], ['7_7', '8_8', '9_9']]   #45
            name_list_na = [['9_1', '8_2', '7_3'], ['6_4', '5_5', '4_6'], ['3_7', '2_8', '1_9']]    #135
            name_list_sel = [
                name_list_heng,
                name_list_shu,
                name_list_pie,
                name_list_na
            ]

        if config.db_name == 'NBU':
            name_list_heng = [['004_000', '004_001', '004_002'], ['004_003', '004_004', '004_005'], # 0
                              ['004_006', '004_007', '004_008']]  
            name_list_shu = [['000_004', '001_004', '002_004'], ['003_004', '004_004', '005_004'],  #90
                             ['006_004', '007_004', '008_004']]
            name_list_pie = [['000_000', '001_001', '002_002'], ['003_003', '004_004', '005_005'],  #45
                             ['006_006', '007_007', '008_008']]
            name_list_na = [['008_000', '007_001', '006_002'], ['005_003', '004_004', '003_005'],    #135
                            ['002_006', '001_007', '000_008']]
            name_list_sel = [
                name_list_heng,
                name_list_shu,
                name_list_pie,
                name_list_na
            ]


        if config.db_name == 'MPI':
            name_list_1 = [['Frame_000', 'Frame_001', 'Frame_002'], ['Frame_003', 'Frame_004', 'Frame_005'],
                           ['Frame_006', 'Frame_007', 'Frame_008']]
            name_list_3 = [['Frame_100', 'Frame_099', 'Frame_098'], ['Frame_097', 'Frame_096', 'Frame_095'],
                           ['Frame_094', 'Frame_093', 'Frame_092']]
            name_list_sel = [name_list_1, name_list_3]

        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                idx, dis, score = line.split()
                idx = int(idx)
                score = float(score)
                if idx in self.scene_list:
                    for aug_num in range(self.aug_num):
                        for i in range(len(name_list_sel)):
                            sai_each = []
                            f_cat = []
                            count = 0
                            for j in range(len(name_list_sel[i])):
                                for n in range(len(name_list_sel[i][j])):
                                    each = '{}/{}.png'.format(dis, name_list_sel[i][j][n])
                                    sai_each.append(each)
                                    count += 1
                                if count >= len(name_list_sel[i][j]):
                                    f_cat.append(sai_each)
                                    sai_each = []
                                    count = 0
                            dis_files_data.append(f_cat)
                            idx_data.append(idx)
                            score_data.append(score)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)
        idx_data = np.array(idx_data)
        idx_data = idx_data.reshape(-1, 1)

        self.data_dict = {
            'd_img_list': dis_files_data,
            'score_list': score_data,
            'idx_list': idx_data
        }

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        h, w = self.config.input_size
        top = random.randint(0, h - config.crop_size)
        left = random.randint(0, w - config.crop_size)
        bottom = top + config.crop_size
        right = left + config.crop_size
        if_flip = random.random()
        resize_percent = 0.8
        angle = random.randint(-45,45) #old 60
        cat_all = []
        resize_count = 0

        if config.db_name == 'win5':
            count_sum = count_win5
        if config.db_name == 'NBU':
            count_sum = count_nbu
        if config.db_name == 'MPI':
            count_sum = count_mpi


        for n in range(len(self.data_dict['d_img_list'][idx])):
            dis = []
            for i in range(len(self.data_dict['d_img_list'][idx][n])):
                d_img_name = self.data_dict['d_img_list'][idx][n][i]
                d_img = Image.open(Path(self.config.db_path) / d_img_name).convert("RGB")
                if self.mode == 'train':
                    if if_flip < resize_percent and resize_count < count_sum:
                        d_img = d_img.resize(self.config.new_size)
                        resize_count += 1
                    else:
                        d_img = d_img.resize(self.config.input_size)
                        d_img = d_img.rotate(angle)
                        d_img = d_img.crop((left, top, right, bottom))
                if self.mode == 'test':
                    d_img = d_img.resize(self.config.new_size)
                if self.transform:
                    d_img = self.transform(d_img)
                dis.append(d_img)
            dis = torch.cat(dis, dim=0)
            cat_all.append(dis)

        score = self.data_dict['score_list'][idx]
        idx = self.data_dict['idx_list'][idx]
        sample = {
            'd_img_org': cat_all,
            'score': score,
            'idx': idx
        }

        return sample
