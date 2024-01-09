'''
use this to train and val
'''
import os
import torch
import numpy as np
import logging
import time
import random
import csv
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from datas.mos_txt import dis_list3
from models.GeoLFIQA import GeoLFIQA
from config import config
from datas.pre_data import IQA_datset


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    idx_epoch = []  # ++

    for data in tqdm(train_loader):
        x_d = data['d_img_org']
        x_d = [x.cuda() for x in x_d]
        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
        idx = data['idx'].cuda()
        pred_d = net(x_d)
        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        idx_batch_numpy = idx.data.cpu().numpy()  # ++
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        idx_epoch = np.append(idx_epoch, idx_batch_numpy)  # ++
    path = config.svPath + '/train/{}'.format(config.model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    dataPath = path + '/train_pred_{}.csv'.format(epoch + 1)
    with open(dataPath, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(idx_epoch, pred_epoch, labels_epoch))

    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))
    print('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))
    return ret_loss, rho_s, rho_p

def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        pred_epoch = []
        labels_epoch = []
        idx_epoch = []  # ++
        count, pred_mean, labels_mean, idx_mean = 0, 0, 0, 0

        for data in tqdm(test_loader):
            pred = 0

            x_d = data['d_img_org']
            x_d = [x.cuda() for x in x_d]
            labels = data['score']
            idx = data['idx'].cuda()  # ++
            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
            pred = net(x_d)

            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            if config.if_avg:
                pred = pred.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
                idx = idx.data.cpu().numpy()

                pred_mean += pred
                labels_mean += labels
                idx_mean += idx
                count += 1
                if config.db_name == 'win5' or 'NBU':
                    avg_num = 4
                if config.db_name == 'MPI':
                    avg_num = 2

                if count >= avg_num:
                    pred_mean = pred_mean / count
                    labels_mean = labels_mean / count
                    idx_mean = idx_mean / count

                    pred_epoch = np.append(pred_epoch, pred_mean)
                    labels_epoch = np.append(labels_epoch, labels_mean)
                    idx_epoch = np.append(idx_epoch, idx_mean)

                    count, pred_mean, labels_mean, idx_mean = 0, 0, 0, 0

            else:
                pred_batch_numpy = pred.data.cpu().numpy()
                labels_batch_numpy = labels.data.cpu().numpy()
                idx_batch_numpy = idx.data.cpu().numpy()  # ++
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)
                idx_epoch = np.append(idx_epoch, idx_batch_numpy)  # ++

        path = config.svPath + '/test/{}'.format(config.model_name)
        if not os.path.exists(path):
            os.mkdir(path)
        dataPath = path + '/test_pred_{}.csv'.format(epoch + 1)
        with open(dataPath, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(idx_epoch, pred_epoch, labels_epoch))

        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rmse  = np.sqrt(mean_squared_error(np.squeeze(labels_epoch), np.squeeze(pred_epoch)))

        logging.info(
            'Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} =====RMSE:{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                 rho_p, rmse))
        print('test epoch:{}  =====  loss:{:.4}  =====  SRCC:{:.4}  =====  PLCC:{:.4} =====RMSE:{:.4}'
              .format(epoch + 1, np.mean(losses), rho_s, rho_p, rmse))

        return np.mean(losses), rho_s, rho_p, rmse


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    setup_seed(20)

    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)

    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    if not os.path.exists(config.tensorboard_path):
        os.mkdir(config.tensorboard_path)

    config.snap_path += config.model_name
    config.log_file = config.model_name + config.log_file
    config.tensorboard_path += config.model_name

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

    train_dis, test_dis = dis_list3()
    train_transform = torchvision.transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.5, std=0.5)
    ])
    test_transforms = torchvision.transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.5, std=0.5)
    ])

    train_dataset = IQA_datset(
        config=config,
        scene_list=train_dis,
        transform=train_transform,
        mode='train',
    )
    val_dataset = IQA_datset(
        config=config,
        scene_list=test_dis,
        transform=test_transforms,
        mode='test',
    )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))
    logging.info('train scenes:{}'.format(train_dis))
    logging.info('test scene:{}'.format(test_dis))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )
    net = GeoLFIQA(
        embed_dim=config.embed_dim,
        patch_size=config.patch_size,
        img_size=config.img_size,
    )

    net = net.cuda()

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    for epoch in range(0, config.n_epoch):

        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader)
        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)
        writer.add_scalar('Learning Rate{}', optimizer.param_groups[0]['lr'], epoch)  # add lr

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, rho_s, rho_p,rmse = eval_epoch(config, epoch, net, criterion, val_loader)

            writer.add_scalar("test_loss", loss, epoch)
            writer.add_scalar("test_SRCC", rho_s, epoch)
            writer.add_scalar("test_PLCC", rho_p, epoch)

            logging.info('Eval done...')
            if rho_s > best_srocc or rho_p > best_plcc:
                best_srocc = rho_s
                best_plcc = rho_p
                bset_rmse = rmse
                # save weights
                model_name = "epoch{}.pth".format(epoch + 1)
                model_save_path = os.path.join(config.snap_path, model_name)
                torch.save(net.state_dict(), model_save_path)
                logging.info(
                    'Saving weights and model of epoch{}, SRCC:{:.4}, PLCC:{:.4}, RMSE:{:.4}'.format(epoch + 1, best_srocc, best_plcc, bset_rmse))

        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))

