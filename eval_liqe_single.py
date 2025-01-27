import torch
import torch.nn as nn
import numpy as np
#from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import clip
import random
import time
from MNL_Loss import Fidelity_Loss, loss_m4, Multi_Fidelity_Loss, Fidelity_Loss_distortion
import scipy.stats
from utils import set_dataset_qonly, _preprocess2, _preprocess3, convert_models_to_fp32
import torch.nn.functional as F
from itertools import product
import os
import pickle
from weight_methods import WeightMethods

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr


def cal_plcc_srcc(gt, pred):
    pred = fit_curve(pred, gt)
    plcc = calculate_plcc(pred, gt)
    srcc = calculate_srcc(pred, gt)
    return plcc, srcc


def calculate_srcc(pred, mos):
    srcc, _ = spearmanr(pred, mos)
    return srcc


def calculate_plcc(pred, mos):
    plcc, _ = pearsonr(pred, mos)
    return plcc


def fit_curve(x, y, curve_type="logistic_4params"):
    r"""Fit the scale of predict scores to MOS scores using logistic regression suggested by VQEG.
    The function with 4 params is more commonly used.
    The 5 params function takes from DBCNN:
        - https://github.com/zwx8981/DBCNN/blob/master/dbcnn/tools/verify_performance.m
    """
    assert curve_type in [
        "logistic_4params",
        "logistic_5params",
    ], f"curve type should be in [logistic_4params, logistic_5params], but got {curve_type}."

    betas_init_4params = [np.max(y), np.min(y), np.mean(x), np.std(x) / 4.0]

    def logistic_4params(x, beta1, beta2, beta3, beta4):
        yhat = (beta1 - beta2) / (1 + np.exp(-(x - beta3) / beta4)) + beta2
        return yhat

    betas_init_5params = [10, 0, np.mean(y), 0.1, 0.1]

    def logistic_5params(x, beta1, beta2, beta3, beta4, beta5):
        logistic_part = 0.5 - 1.0 / (1 + np.exp(beta2 * (x - beta3)))
        yhat = beta1 * logistic_part + beta4 * x + beta5
        return yhat

    if curve_type == "logistic_4params":
        logistic = logistic_4params
        betas_init = betas_init_4params
    elif curve_type == "logistic_5params":
        logistic = logistic_5params
        betas_init = betas_init_5params

    betas, _ = curve_fit(logistic, x, y, p0=betas_init, maxfev=10000)
    yhat = logistic(x, *betas)
    return yhat


def do_batch(x, text):
    batch_size = x.size(0)
    num_patch = x.size(1)

    x = x.view(-1, x.size(2), x.size(3), x.size(4))

    logits_per_image, logits_per_text = model.forward(x, text)

    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_text = logits_per_text.view(-1, batch_size, num_patch)

    logits_per_image = logits_per_image.mean(1)
    logits_per_text = logits_per_text.mean(2)

    logits_per_image = F.softmax(logits_per_image, dim=1)

    return logits_per_image, logits_per_text


##############################textual template####################################
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

##############################general setup####################################
img_dir = "/data/zyyou/DataDepictQA"
seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

initial_lr = 5e-6
num_epoch = 6
bs = 64

train_patch = 3

loss_img2 = Fidelity_Loss_distortion()
loss_scene = Multi_Fidelity_Loss()

joint_texts = torch.cat([clip.tokenize(f"a photo with {c} quality") for c in qualitys]).to(device)

##############################general setup####################################

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()


def eval(loader):
    model.eval()
    q_mos = []
    q_hat = []
    for step, sample_batched in enumerate(loader, 0):

        x, gmos = sample_batched['I'], sample_batched['mos']

        x = x.to(device)
        #q_mos.append(gmos.data.numpy())
        q_mos = q_mos + gmos.cpu().tolist()

        # Calculate features
        with torch.no_grad():
            logits_per_image, _ = do_batch(x, joint_texts)

        logits_per_image = logits_per_image.view(-1, len(qualitys))

        logits_quality = logits_per_image

        quality_preds = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                        4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

        q_hat = q_hat + quality_preds.cpu().tolist()

    plcc, srcc = cal_plcc_srcc(q_mos, q_hat)
    return srcc, plcc


if __name__ == "__main__":
    num_workers = 8
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    state_dict = torch.load("/data/zyyou/LIQE/checkpoints/1/liqe_qonly.pt")
    model.load_state_dict(state_dict["model_state_dict"])

    test_csv_files = [
        "/data/zyyou/LIQE/IQA_Database/koniq-10k/test_koniq_2k.txt",
        "/data/zyyou/LIQE/IQA_Database/spaq/test_spaq_2k.txt",
        "/data/zyyou/LIQE/IQA_Database/kadid10k/test_kadid_2k.txt",
        "/data/zyyou/LIQE/IQA_Database/livewild/test_livew_1k.txt",
        "/data/zyyou/LIQE/IQA_Database/agiqa/test_agiqa_3k.txt",
        "/data/zyyou/LIQE/IQA_Database/CSIQ/test_csiq_866.txt",
        "/data/zyyou/LIQE/IQA_Database/tid2013/test_tid2013_3k.txt",
    ]
    for test_csv_file in test_csv_files:
        test_loader = set_dataset_qonly(test_csv_file, 32, img_dir, num_workers, 
                                        preprocess2, 15, True, set=2)
        srcc, plcc = eval(test_loader)
        print("=" * 100)
        print(test_csv_file)
        print(f"SRCC: {srcc}, PLCC: {plcc}")
