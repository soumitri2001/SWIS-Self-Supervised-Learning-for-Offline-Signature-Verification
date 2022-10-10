import os
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt

from utils import *
from loss import *
from model import *
from optimizer import *
from dataset import get_dataloader_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    dl = get_dataloader_train(args, ben_train_df)

    model = Model(224, 224, args.batchsize, ptsz=args.ptsz, pout=512).cuda()
    loss_fn = NewLossFn(args.batchsize)

    params, param_names = [], []
    for name, param in model.named_parameters():
        params.append(param)
        param_names.append(name)
    parameters = [{'params' : params, 'param_names' : param_names}]
    optim = LARS(parameters, lr = 0.1, weight_decay = 0.9, exclude_from_weight_decay=["batch_normalization", "bias"])

    scheduler = LinearWarmupCosineAnnealingLR(optim, 10, 1000)

    torch.autograd.set_detect_anomaly(True)
    losses = []
    stime = time.time()
    for i in range(args.epochs):
        print('Epoch: ',i+1)
        stime = time.time()
        lossval = 0
        for x1, x2 in dl:
            #print(x1.shape)
            optim.zero_grad()
            (z1, z1so), (z2, z2so) = model(x1.float().cuda()), model(x2.float().cuda())
            loss = loss_fn(z1, z2, z1so, z2so)
            loss.backward()
            optim.step()

            lossval += loss.cpu().detach().numpy()
            #print(loss.detach().numpy())
        scheduler.step()

        lossval = lossval/len(dl)
        losses.append(lossval)
        print('Loss: ',lossval)
        print("Time Taken for Training: ",(time.time()-stime)/60," minutes")
    plt.plot(losses)

    torch.save({'model_state_dict' : model.state_dict(),
                'optim_state_dict' : optim.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'epochs' : i+1},
                f'/content/outputs/model_{args.expt_name}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/content/DATASETS/BHSig260/Bengali', help='path/to/data')
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize used')
    parser.add_argument('--epochs', type=int, default=200, help='epochs of training')
    parser.add_argument('--ptsz', type=int, default=32, help='patch dimensions')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--expt_name', type=str, default='bhsig_bengali_newloss', help='name of experiment')
    args = parser.parse_args()

    main(args)