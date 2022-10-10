import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.svm import SVC as SVM

from model import *
from utils import *
from dataset import get_dataloader_ds

def main(args):

    seed_everything(42)

    ben_train_path = os.path.join(args.datapath, 'TrainSet')
    if not os.path.exists(ben_train_path):
        os.makedirs(ben_train_path)
    ben_test_path = os.path.join(args.datapath, 'TestSet')
    if not os.path.exists(ben_test_path):
        os.makedirs(ben_test_path)
    
    ben_fols = os.listdir(args.datapath)
    ben_writers = []
    for fol in ben_fols:
        if not (fol.endswith('.forgery') or fol.endswith('.genuine') or fol.endswith('.txt') or fol == 'TrainSet' or fol == 'TestSet'):
            ben_writers.append(fol)

    train_writers = random.sample(ben_writers, k = 100)
    test_writers = [w for w in ben_writers if w not in train_writers]

    ben_train_df = pd.DataFrame(columns = ['filepath', 'writer', 'label'])
    ben_test_df = pd.DataFrame(columns = ['filepath', 'writer', 'label'])

    for fol in train_writers:
        files = os.listdir(os.path.join(args.datapath, fol))
        for f in files:
            ben_train_df = ben_train_df.append({'filepath':os.path.join(args.datapath, fol, f), 'writer': fol, 'label': 1 if 'G' in f else 0}, ignore_index = True)
    for fol in test_writers:
        files = os.listdir(os.path.join(args.datapath, fol))
        for f in files:
            ben_test_df = ben_test_df.append({'filepath':os.path.join(args.datapath, fol, f), 'writer': fol, 'label': 1 if 'G' in f else 0}, ignore_index = True)

    ds_train_df = pd.DataFrame(columns = ['filepath', 'writer', 'label'])
    ds_test_df = pd.DataFrame(columns = ['filepath', 'writer', 'label'])
    train_num = dict(zip(test_writers, [8]*len(test_writers)))
    #val_num = 4
    for i in range(len(ben_test_df)):    
        filepath = ben_test_df.iloc[i]['filepath']
        writer = ben_test_df.iloc[i]['writer']
        label = ben_test_df.iloc[i]['label']
        if 'G' in filepath and train_num[writer] > 0:
            ds_train_df = ds_train_df.append({'filepath':filepath, 'writer':writer, 'label':label}, ignore_index = True)
            train_num[writer] -= 1
        #elif 'G' in filepath and train_num == 0 and val_num > 0:
        #    ds_val_df = ds_val_df.append({'filepath':filepath, 'writer':writer, 'label':label}, ignore_index = True)
        #    val_num -= 1
        else:
            ds_test_df = ds_test_df.append({'filepath':filepath, 'writer':writer, 'label':label}, ignore_index = True)

    tdl, vdl = get_dataloader_ds(args, ds_train_df, ds_test_df)

    model = Model(224, 224, args.batchsize, ptsz=args.ptsz, pout=512).cuda() # make sure the model defined here matches with the saved model to be loaded
    model.load_state_dict(torch.load(args.load_model)['model_state_dict'])

    dsmodel = model #.base_encoder #DSModel(model).cuda() #pout=512 for ResNet18 and 2048 for ResNet50
    dsmodel.proj2 = nn.Identity()
    dsmodel.bs = 1

    dsmodel.eval()

    #### feature extraction ####
    for i in range(1):
        accs = []
        losses = []
        aucs = []
        preds = {} #np.array([]).reshape((0,1))
        gt = np.array([]).reshape((0,1))
        with torch.no_grad():
            #accuracy = 0
            for x, y, w in tdl:
                #print(x.shape)
                z1,_ = dsmodel(x.float().cuda())
                z1 = z1.cpu().detach().numpy().reshape((-1,512))
                try:
                    #print(w.numpy(), preds[w.numpy()[0]])
                    preds[w[0]] = np.append(preds[w[0]], z1, axis = 0)
                except:
                    preds[w[0]] = np.array([]).reshape((0,512))
                    preds[w[0]] = np.append(preds[w[0]], z1, axis = 0)

    predsall = preds
    label_dict = {}
    l = 0
    X = np.array([]).reshape((0,512))
    Y = np.array([]).reshape((0,1))
    for k in list(predsall.keys()):
        xs = predsall[k]
        X = np.append(X, xs, axis = 0)
        Y = np.append(Y, np.array([l]*xs.shape[0]).reshape((-1, 1)), axis = 0)
        label_dict[l] = k
        l+=1   

    #### evaluation ####
    predq = np.array([]).reshape((0,512))
    gt = np.array([])#.reshape((0,1))
    gty = np.array([])
    for x, y, w in vdl:
        z1,_ = dsmodel(x.float().cuda())
        z1 = z1.cpu().detach().numpy()
        predq = np.append(predq, z1.reshape((1,512)), axis = 0)
        gt = np.append(gt, w)
        gty = np.append(gty, y)

    #### classification using SVM ####
    clf = SVM(C = 0.5, kernel = 'rbf')
    clf.fit(X, Y.reshape(-1))
    preds = clf.predict(predq)
    predy = np.array([])
    accuracy = 0
    for i in range(preds.shape[0]):
        print(label_dict[preds[i]], gt[i])
        if label_dict[preds[i]]==gt[i] and gty[i] == 1:
            predy = np.append(predy, np.array([1]))
            accuracy+=1
        elif label_dict[preds[i]]!=gt[i] and gty[i] == 0:
            predy = np.append(predy, np.array([0]))
            accuracy+=1
        elif label_dict[preds[i]]!=gt[i] and gty[i] == 1:
            predy = np.append(predy, np.array([0]))
        elif label_dict[preds[i]]==gt[i] and gty[i] != 1:
            predy = np.append(predy, np.array([1]))
            #predy = np.append(predy, np.array([0]))
    print(accuracy/preds.shape[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/content/DATASETS/BHSig260/Bengali', help='path/to/data')
    parser.add_argument('--batchsize', type=int, default=1, help='batchsize used')
    parser.add_argument('--ptsz', type=int, default=32, help='patch dimensions')
    parser.add_argument('--load_model', type=str, default='/content/model_bhsig_bengali_newloss.pt', help='path to model to be loaded')
    args = parser.parse_args()

    main(args)