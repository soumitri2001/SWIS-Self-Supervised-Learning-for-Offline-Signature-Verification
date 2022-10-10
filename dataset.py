import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import imageio
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu


# SSL pretrain dataset
class DatasetSSL(Dataset):
    def __init__(self, data_path = None, df = None, ptsz = 32):
        super(DatasetSSL, self).__init__()

        self.data_path = data_path
        self.transformations = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.2, 0.2)],p = 0.5),
                                                   transforms.RandomApply([transforms.RandomAffine(5, (0.1,0.1), (1.0,1.25))], p=0.2),
                                                   transforms.RandomResizedCrop(224, scale = (0.9,1.0)),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                   ])
        self.df = df
        self.ptsz = ptsz

    def __len__(self):
        return len(self.df)

    def __getpatches__(self, x):
        pts = []
        #print(x.shape)
        #x = resize(x, (224,224), preserve_range = True)
        #print(x.shape)
        H,W,C = x.shape
        numdelH = 224//(self.ptsz//2) - 1
        numdelW = 224//(self.ptsz//2) - 1

        for i in range(numdelH):
            for j in range(numdelW):
                sx = i*(self.ptsz//2)
                ex = sx + self.ptsz
                sy = j*(self.ptsz//2)
                ey = sy + self.ptsz
                #print(sx,ex,sy,ey)
                temp = x[sx:ex,sy:ey,:]
                temp = np.transpose(temp, (2,0,1))
                temp = torch.from_numpy(temp)
                #print(temp.shape)        
                pts.append(torch.unsqueeze(temp, 0))

        return torch.cat(pts, dim = 0)

    def __get_com_cropped__(self, image):
        '''image is a binary PIL image'''
        thresh = threshold_otsu(image[:,:,0])
        image2 = image > thresh
        #plt.imshow(np.asarray(image2).astype(np.float))
        com = ndimage.measurements.center_of_mass(image2)
        com = np.round(com)
        com[0] = np.clip(com[0], 0, image.shape[0])
        com[1] = np.clip(com[1], 0, image.shape[1])
        X_center, Y_center = int(com[0]), int(com[1])
        c_row, c_col = image[X_center, :, 0], image[:, Y_center, 0]

        x_start, x_end, y_start, y_end = -1, -1, -1, -1

        for i, v in enumerate(c_col):
            v = np.sum(image2[i, :, 0])
            if v < image.shape[1]: # there exists text pixel
                if x_start == -1:
                    x_start = i
                else:
                    x_end = i

        for j, v in enumerate(c_row):
            v = np.sum(image2[:, j, 0])
            if v < image.shape[0]: # there exists text pixel
                if y_start == -1:
                    y_start = j
                else:
                    y_end = j
        
        crop_rgb = image[x_start:x_end, y_start:y_end, :] #)).convert('RGB')
        return crop_rgb

    def __augment__(self,x):
        x1, x2 = self.transformations(x), self.transformations(x)
        return x1, x2

    def __getitem__(self, idx):
        #writer = self.df['writer'].iloc[idx]

        orgpic = cv.imread(self.df.iloc[idx]['filepath'])
        orgpic = orgpic/255.0
        orgpic = self.__get_com_cropped__(orgpic)
        #plt.imshow(orgpic)

        orgpic = resize(orgpic, (224,224), preserve_range = True)
        orgpic = np.transpose(orgpic, (2,0,1))
        orgpic = torch.from_numpy(orgpic).float()

        orgpic1, orgpic2 = self.__augment__(orgpic)
        orgpic1 = orgpic1.numpy().transpose(1,2,0)
        orgpic2 = orgpic2.numpy().transpose(1,2,0)

        orgpic1pts = self.__getpatches__(orgpic1)
        orgpic2pts = self.__getpatches__(orgpic2)
        
        #print(orgpic1pts.shape)
        return orgpic1pts, orgpic2pts 


def get_dataloader_train(args, train_df):
    ds = DatasetSSL(data_path=args.datapath, df=train_df, ptsz=args.ptsz)
    dl = DataLoader(ds, batch_size=args.batchsize, drop_last=True, shuffle=True)
    return dl


# downstream dataset
class DatasetDownstream(Dataset):
    def __init__(self, df=None, ptsz=32):
        super(DatasetDownstream, self).__init__()
        self.df = df
        self.ptsz = ptsz
        self.centercrop = transforms.CenterCrop(224)
    
    def __len__(self):
        return len(self.df)

    def __getpatches__(self, x):
        pts = []
        #print(x.shape)
        
        #x = self.centercrop(torch.from_numpy(x)).numpy()
        #print(x.shape)
        H,W,C = x.shape
        numdelH = 224//(self.ptsz//2) - 1
        numdelW = 224//(self.ptsz//2) - 1

        for i in range(numdelH):
            for j in range(numdelW):
                sx = i*(self.ptsz//2) 
                ex = sx + self.ptsz
                sy = j*(self.ptsz//2)
                ey = sy + self.ptsz
                #print(sx,ex,sy,ey)
                temp = x[:,sx:ex,sy:ey]
                #temp = np.transpose(temp, (2,0,1))
                temp = torch.from_numpy(temp)
                #print(temp.shape)        
                pts.append(torch.unsqueeze(temp, 0))

        return torch.cat(pts, dim = 0)

    def __get_com_cropped__(self, image):
        '''image is a binary PIL image'''
        thresh = threshold_otsu(image[:,:,0])
        image2 = image > thresh
        #plt.imshow(np.asarray(image2).astype(np.float))
        com = ndimage.measurements.center_of_mass(image2)
        com = np.round(com)
        com[0] = np.clip(com[0], 0, image.shape[0])
        com[1] = np.clip(com[1], 0, image.shape[1])
        X_center, Y_center = int(com[0]), int(com[1])
        c_row, c_col = image[X_center, :, 0], image[:, Y_center, 0]

        x_start, x_end, y_start, y_end = -1, -1, -1, -1

        for i, v in enumerate(c_col):
            v = np.sum(image2[i, :, 0])
            if v < image.shape[1]: # there exists text pixel
                if x_start == -1:
                    x_start = i
                else:
                    x_end = i

        for j, v in enumerate(c_row):
            v = np.sum(image2[:, j, 0])
            if v < image.shape[0]: # there exists text pixel
                if y_start == -1:
                    y_start = j
                else:
                    y_end = j
        
        #print(x_start, y_start, x_end, y_end)

        crop_rgb = image[x_start:x_end, y_start:y_end, :] #)).convert('RGB')
        return crop_rgb

    def __getitem__(self, idx):
        
        pic = cv.imread(self.df.iloc[idx]['filepath'])
        #pic = 1 - pic #np.transpose(pic, (2,0,1))
        x = resize(pic, (256,256), preserve_range = True)
        #x = x[32:x.shape[0]-32,32:x.shape[1]-32,:]
        #print(x.shape)
        #x = np.transpose(x, (2,0,1))
        x = x/255.0
        x = self.__get_com_cropped__(x)
        x = resize(x, (224,224), preserve_range = True)
        #x = self.centercrop(torch.from_numpy(x)).numpy()
        #x = self.__get_com_cropped__(x)
        #x = resize(x, (224,224), preserve_range = True)
        x = (-0.5 + x)/0.5
        x = np.transpose(x, (2,0,1))


        writer = self.df.iloc[idx]['writer']

        picpts = self.__getpatches__(x)
        
        return picpts, self.df['label'].iloc[idx], writer


def get_dataloader_ds(args, train_df, test_df):
    tds, vds = DatasetDownstream(train_df, args.ptsz), DatasetDownstream(test_df, args.ptsz)
    tdl = torch.utils.data.DataLoader(tds, batch_size=1, drop_last=True, shuffle=True)
    vdl = torch.utils.data.DataLoader(vds, batch_size=1, drop_last=True, shuffle=True)
    return tdl, vdl