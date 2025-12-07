import logging
import os
import random
import sys
import math
import numpy as np
import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import utils
import rasterio
from rasterio. windows import Window
from rasterio.errors import RasterioError
from torch.utils.data.dataset import IterableDataset
from copy import deepcopy

# 导入自定义损失函数
from losses_custom import ComboLoss, make_pseudo_label


class StreamingGeospatialDataset(IterableDataset):
    
    def __init__(self, imagery_fns, label_fns=None, groups=None, chip_size=256, num_chips_per_tile=200, windowed_sampling=False, image_transform=None, label_transform=None, nodata_check=None, verbose=False):
        if label_fns is None:
            self.fns = imagery_fns
            self.use_labels = False
        else:
            self.fns = list(zip(imagery_fns, label_fns)) 
            
            self.use_labels = True

        self.groups = groups

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.windowed_sampling = windowed_sampling

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.nodata_check = nodata_check

        self.verbose = verbose

        if self.verbose:
            print("Constructed StreamingGeospatialDataset")

    def stream_tile_fns(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # In this case we are not loading through a DataLoader with multiple workers
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # We only want to shuffle the order we traverse the files if we are the first worker (else, every worker will shuffle the files...)
        if worker_id == 0:
            np.random. shuffle(self.fns) # in place

        if self.verbose:
            print("Creating a filename stream for worker %d" % (worker_id))

        # This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from.  If the number of workers is greater than the number of filenames, some workers will not receive any filenames.
        N = len(self.fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id+1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):

            label_fn = None
            if self.use_labels:
                
                img_fn, label_fn = self.fns[idx]
            else:
                img_fn = self.fns[idx]

            if self.groups is not None:
                group = self.groups[idx]
            else:
                group = None

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (img_fn, label_fn, group)

    def stream_chips(self):
        for img_fn, label_fn, group in self.stream_tile_fns():
            num_skipped_chips = 0

            # Open file pointers
            img_fp = rasterio.open(img_fn, "r")
            label_fp = rasterio.open(label_fn, "r") if self.use_labels else None
            
            height, width = img_fp.shape
            if self.use_labels: # garuntee that our label mask has the same dimensions as our imagery
                t_height, t_width = label_fp. shape
                assert height == t_height and width == t_width

            # If we aren't in windowed sampling mode then we should read the entire tile up front
            img_data = None
            label_data = None
            try:
                if not self.windowed_sampling:
                    img_data = np.rollaxis(img_fp.read(3), 0, 3)
                    if self.use_labels:
                        label_data = label_fp.read(). squeeze() # assume the label geotiff has a single channel
            except RasterioError as e:
                print("WARNING: Error reading in entire file, skipping to the next file")
                continue

            for i in range(self.num_chips_per_tile):
                # Select the top left pixel of our chip randomly 
                x = np.random.randint(0, width-self.chip_size)
                y = np.random.randint(0, height-self.chip_size)

                # Read imagery / labels
                img = None
                labels = None
                if self.windowed_sampling:
                    try:
                        img = np.rollaxis(img_fp.read(window=Window(x, y, self.chip_size, self.chip_size)), 0, 3)
                        # print(img.shape)
                        if self.use_labels:
                            labels = label_fp. read(window=Window(x, y, self.chip_size, self. chip_size)).squeeze()
                    except RasterioError:
                        print("WARNING: Error reading chip from file, skipping to the next chip")
                        continue
                else:
                    img = img_data[y:y+self.chip_size, x:x+self.chip_size, :]
                    if self. use_labels:
                        labels = label_data[y:y+self.chip_size, x:x+self.chip_size]

                # Check for no data
                if self.nodata_check is not None:
                    if self.use_labels:
                        skip_chip = self.nodata_check(img, labels)
                    else:
                        skip_chip = self.nodata_check(img)

                    if skip_chip: # The current chip has been identified as invalid by the `nodata_check(...)` method
                        num_skipped_chips += 1
                        continue

                # 数据增强：随机翻转 + 随机旋转90度
                if self.use_labels:
                    img, labels = self._augment(img, labels)
                else:
                    img = self._augment_img_only(img)

                # Transform the imagery
                if self.image_transform is not None:
                    if self.groups is None:
                        img = self. image_transform(img)
                    else:
                        img = self.image_transform(img, group)
                else:
                    img = torch.from_numpy(img).squeeze()

                # Transform the labels
                if self.use_labels:
                    if self.label_transform is not None:
                        if self.groups is None:
                            
                            labels = self.label_transform(labels)
                        else:
                            print(label_fn)
                            labels = self.label_transform(labels, group)
                            print(labels)
                    else:
                        labels = torch.from_numpy(labels).squeeze()

                # Note, that img should be a torch "Double" type (i.e. a np.float32) and labels should be a torch "Long" type (i.e. np.int64)
                if self.use_labels:
                     yield img, labels
                else:
                     yield img
            # Close file pointers
            img_fp.close()
            if self.use_labels:
                label_fp.close()

            if num_skipped_chips>0 and self.verbose:
                print("We skipped %d chips on %s" % (num_skipped_chips, img_fn))

    def _augment(self, img, labels):
        """同步几何增强：随机水平翻转 + 随机90度旋转"""
        # 随机水平翻转
        if random.random() < 0.5:
            img = np.flip(img, axis=1). copy()
            labels = np.flip(labels, axis=1).copy()
        # 随机垂直翻转
        if random.random() < 0.5:
            img = np. flip(img, axis=0). copy()
            labels = np. flip(labels, axis=0). copy()
        # 随机旋转90度（0, 90, 180, 270）
        k = random.randint(0, 3)
        img = np.rot90(img, k, axes=(0, 1)). copy()
        labels = np. rot90(labels, k, axes=(0, 1)).copy()
        return img, labels

    def _augment_img_only(self, img):
        """仅图像几何增强"""
        if random.random() < 0.5:
            img = np. flip(img, axis=1). copy()
        if random.random() < 0.5:
            img = np. flip(img, axis=0). copy()
        k = random.randint(0, 3)
        img = np.rot90(img, k, axes=(0, 1)).copy()
        return img

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())


def image_transforms(img):
    img = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
    img = np.rollaxis(img, 2, 0). astype(np.float32)
    img = torch.from_numpy(img)
    return img


def label_transforms(labels):
    labels = utils. LABEL_CLASS_TO_IDX_MAP[labels]
    labels = torch.from_numpy(labels)
    return labels


def nodata_check(img, labels):
    return np.any(labels == 0) or np.any(np.sum(img == 0, axis=2) == 4)


def ema_update(src_model, tgt_model, decay):
    """EMA模型更新"""
    with torch.no_grad():
        msd = src_model.state_dict()
        for k, v in tgt_model.state_dict().items():
            v.copy_(v * decay + msd[k] * (1 - decay))


def trainer_dataset(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging. INFO,
                        format='[%(asctime)s. %(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size

    #-------------------
    # Load input data
    #-------------------
    
    input_dataframe = pd.read_csv(args.list_dir)
    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values
    NUM_CHIPS_PER_TILE = 50  # How many chips will be sampled from one large-scale tile 
    CHIP_SIZE = 224  # Size of each sampled chip
    db_train = StreamingGeospatialDataset(
        imagery_fns=image_fns, label_fns=label_fns, groups=None, chip_size=CHIP_SIZE, num_chips_per_tile=NUM_CHIPS_PER_TILE, windowed_sampling=True, verbose=False,
        image_transform=image_transforms, label_transform=label_transforms, nodata_check=nodata_check
    )

    print("The length of train set is: {}".format(len(image_fns)*NUM_CHIPS_PER_TILE))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model.train()
    
    # 使用新的组合损失（CE+标签平滑+Dice），ignore_index=255
    criterion_main = ComboLoss(ce_eps=0.1, dice_w=0.5, ignore_index=255)
    criterion_aux = ComboLoss(ce_eps=0.1, dice_w=0.3, ignore_index=255)
    
    # 使用 AdamW 优化器
    optimizer = AdamW(model. parameters(), lr=base_lr, weight_decay=0.01)
    
    # 学习率调度：warmup + cosine annealing
    num_training_batches_per_epoch = int(len(image_fns) * NUM_CHIPS_PER_TILE / batch_size)
    warmup_steps = 1000
    total_steps = args.max_epochs * num_training_batches_per_epoch
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=base_lr * 0.05)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda s: min(1.0, (s + 1) / warmup_steps))
    
    # EMA 模型
    ema_decay = 0.999
    model_ema = deepcopy(model). cuda()
    for p in model_ema.parameters():
        p.requires_grad_(False)
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(image_fns) * NUM_CHIPS_PER_TILE
    logging.info("{} iterations per epoch.  {} max iterations ". format(len(image_fns)*NUM_CHIPS_PER_TILE, max_iterations))
    iterator = range(max_epoch)
    
    for epoch_num in iterator:
        loss1 = []
        loss2 = []
        for i_batch, (image_batch, label_batch) in tqdm(enumerate(trainloader), total=num_training_batches_per_epoch):
            image_batch, label_batch = image_batch. cuda(), label_batch.cuda()
            outputs1, outputs2 = model(image_batch)
            
            # 生成置信过滤伪标签（供 branch2 使用）
            pseudo = make_pseudo_label(outputs1, label_batch, conf_th=0.7, ignore_index=255)
            
            # 计算损失
            loss_ce1 = criterion_main(outputs1, label_batch. long())  # 主分支损失
            loss_ce2 = criterion_aux(outputs2, pseudo.long())        # 辅分支使用伪标签
            loss = 0.5 * loss_ce1 + 0.5 * loss_ce2

            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 学习率调度：先 warmup，再 cosine
            if iter_num < warmup_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler. step()
            
            # EMA 更新
            ema_update(model, model_ema, ema_decay)
            
            loss1.append(loss_ce1.item())
            loss2. append(loss_ce2.item())
            iter_num = iter_num + 1
            
        avg_loss1 = np.mean(loss1)
        avg_loss2 = np.mean(loss2)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch : %d, CE-branch1 : %f, MCE-branch2: %f, loss: %f, lr: %e' % (epoch_num, avg_loss1, avg_loss2, avg_loss1*0.5+avg_loss2*0.5, current_lr))
        
        save_interval = 20 
        if epoch_num % save_interval == 0:
            # 保存普通模型
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging. info("save model to {}".format(save_mode_path))
            # 保存 EMA 模型
            save_ema_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '_ema.pth')
            torch.save(model_ema.state_dict(), save_ema_path)
            logging.info("save EMA model to {}".format(save_ema_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model. state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            save_ema_path = os.path. join(snapshot_path, 'epoch_' + str(epoch_num) + '_ema.pth')
            torch.save(model_ema.state_dict(), save_ema_path)
            logging.info("save EMA model to {}".format(save_ema_path))
            break

    writer.close()
    return "Training Finished!"
