import clip
import numpy as np
import os
import pickle
import random
import scipy.stats
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.optim import lr_scheduler
from utils_rag import set_dataset, _preprocess2, _preprocess3, convert_models_to_fp32


class AlignModel(nn.Module):

    def __init__(self):
        super(AlignModel, self).__init__()
        dim_text = 384
        dim_visual = 512
        self.temperature = 0.2
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.model_visual = model.visual
        self.model_text = SentenceTransformer("all-MiniLM-L6-v2")
        self.proj = nn.Linear(dim_text, dim_visual)
        self.head = nn.Linear(2 * dim_visual, 1)

    def forward(self, img, text):
        batch_size, num_patch = img.shape[:2]
        img = img.view(-1, img.size(2), img.size(3), img.size(4))
        emb_img = self.model_visual(img)  # [B x N, 512]
        emb_text = self.model_text.encode(text, convert_to_tensor=True)  # [B, 384]
        emb_text = self.proj(emb_text)  # [B, 512]

        emb_img_head = emb_img.view(batch_size, num_patch, -1)
        emb_text_head = emb_text.unsqueeze(1).repeat(1, num_patch, 1)
        pred = self.head(torch.cat([emb_img_head, emb_text_head], dim=2))
        pred = pred.view(batch_size, num_patch, -1).mean(dim=1)

        cosine_similarity = F.cosine_similarity(
            emb_img.unsqueeze(1), emb_text.unsqueeze(0), dim=2
        )  # [B x N, B]
        # mean in num_patch dimension
        cosine_similarity = cosine_similarity.view(batch_size, num_patch, batch_size)
        cosine_similarity = cosine_similarity.mean(1)  # [B, B]
        cosine_similarity = cosine_similarity / self.temperature
        sim_per_img = F.softmax(cosine_similarity, dim=1)
        sim_per_text = F.softmax(cosine_similarity, dim=0)
        loss_per_img = -sim_per_img.diagonal().log().mean()
        loss_per_text = -sim_per_text.diagonal().log().mean()
        loss_nce = loss_per_img + loss_per_text

        return loss_nce, pred


def cal_fidelity_loss(pred_A, pred_B, gmos_A, gmos_B):
    eps = 1e-8
    gt = (gmos_A > gmos_B).float().detach()
    pred = 0.5 * (1 + torch.erf((pred_A - pred_B) / 2))  # 2 -> sqrt(2 * (1**2 + 1**2))
    loss = (1 - (pred * gt + eps).sqrt() - ((1 - pred) * (1 - gt) + eps).sqrt()).mean()
    return loss


##############################general setup####################################
save_dir = "2_nce+fidelity"
img_dir = "/root/Data4ICCV"
seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

initial_lr = 5e-6
num_epoch = 10
num_steps_per_epoch = 1000
bs = 12
train_patch = 3
weight_score = 20.


##############################general setup####################################

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()

opt = 0
def freeze_model(opt):
    if opt == 0: # do nothing
        return
    elif opt == 1: # freeze text encoder
        for p in model.model_text.parameters():
            p.requires_grad = False
    elif opt == 2: # freeze visual encoder
        for p in model.model_visual.parameters():
            p.requires_grad = False
    elif opt == 3:
        for p in model.parameters():
            p.requires_grad = False


def train(model, best_result, best_epoch, srcc_dict):
    start_time = time.time()
    beta = 0.9
    running_loss = 0 if epoch == 0 else train_loss[-1]
    running_duration = 0.0
    local_counter = epoch * num_steps_per_epoch + 1
    convert_models_to_fp32(model)
    model.eval()

    loaders = []
    for loader in train_loaders:
        loaders.append(iter(loader))

    print(optimizer.state_dict()['param_groups'][0]['lr'])
    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
    for step in range(num_steps_per_epoch):
        #total_loss = 0
        I_A_batch, text_A_batch, gmos_A_batch = [], [], []
        I_B_batch, text_B_batch, gmos_B_batch = [], [], []

        for dataset_idx, loader in enumerate(loaders, 0):
            try:
                sample_batched = next(loader)
            except StopIteration:
                loader = iter(train_loaders[dataset_idx])
                sample_batched = next(loader)
                loaders[dataset_idx] = loader

            I_A, text_A, gmos_A = sample_batched['I_A'], sample_batched["text_A"], sample_batched['mos_A']
            I_A_batch.append(I_A.to(device))
            text_A_batch += text_A
            gmos_A_batch.append(gmos_A.to(device))
            I_B, text_B, gmos_B = sample_batched['I_B'], sample_batched["text_B"], sample_batched['mos_B']
            I_B_batch.append(I_B.to(device))
            text_B_batch += text_B
            gmos_B_batch.append(gmos_B.to(device))

        I_A_batch = torch.cat(I_A_batch, dim=0)  # [128, 3, 3, 224, 224]
        gmos_A_batch = torch.cat(gmos_A_batch, dim=0)  # [128, ]
        I_B_batch = torch.cat(I_B_batch, dim=0)  # [128, 3, 3, 224, 224]
        gmos_B_batch = torch.cat(gmos_B_batch, dim=0)  # [128, ]

        loss_nce_A, pred_A = model(I_A_batch, text_A_batch)
        loss_nce_B, pred_B = model(I_B_batch, text_B_batch)
        loss_score = cal_fidelity_loss(pred_A, pred_B, gmos_A_batch, gmos_B_batch)
        loss = (loss_nce_A + loss_nce_B) + weight_score * loss_score

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        running_loss = beta * running_loss + (1 - beta) * loss.data.item()
        loss_corrected = running_loss / (1 - beta ** local_counter)

        current_time = time.time()
        duration = current_time - start_time
        running_duration = beta * running_duration + (1 - beta) * duration
        duration_corrected = running_duration / (1 - beta ** local_counter)
        examples_per_sec = I_A.size(0) / duration_corrected
        format_str = (
            "(Epoch: %d, Step: %d / %d)"
            "[Running Loss = %.4f] [Loss NCE A = %.4f] [Loss NCE B = %.4f] [Loss Score = %.4f] "
            "(%.1f samples/sec; %.3f sec/batch)"
        )
        print(
            format_str % (
                epoch, step + 1, num_steps_per_epoch, loss_corrected, 
                loss_nce_A, loss_nce_B, loss_score, examples_per_sec, duration_corrected
            )
        )

        local_counter += 1
        start_time = time.time()

        train_loss.append(loss_corrected)

    all_result = {'val':{}, 'test':{}}
    if (epoch >= 0):
        srcc = eval(test_loader, phase='val', dataset='koniq10k')
        print('**********New overall results!**********')
        best_epoch = epoch
        best_result = srcc
        srcc_dict['koniq10k'] = srcc

        os.makedirs(os.path.join(f"checkpoints/{save_dir}"), exist_ok=True)
        ckpt_name = os.path.join(f"checkpoints/{save_dir}/ckpt.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'all_results': all_result
        }, ckpt_name)  # just change to your preferred folder/filename

    return best_result, best_epoch, srcc_dict, all_result


def eval(test_loader, phase, dataset):
    model.eval()
    q_mos = []
    q_pred = []

    for sample_batched in test_loader:
        I, text, gmos = sample_batched['I_A'], sample_batched["text_A"], sample_batched['mos_A']
        I = I.to(device)
        q_mos = q_mos + gmos.cpu().tolist()
        with torch.no_grad():
            _, pred = model(I, text)
        q_pred = q_pred + pred.squeeze(1).cpu().tolist()

    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_pred)[0]

    print_text = dataset + ' ' + phase + ' finished'
    print(print_text)
    return srcc


num_workers = 8
model = AlignModel().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=initial_lr,
    weight_decay=0.001)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

train_loss = []
start_epoch = 0

freeze_model(opt)

best_result = 0
best_epoch = 0

# avg
srcc_dict = {'koniq10k': 0.0}

spaq_meta = "/root/Data4ICCV/metas/metas_spaq.json"
liveitw_meta = "/root/Data4ICCV/metas/metas_liveitw.json"
livefb_meta = "/root/Data4ICCV/metas/metas_livefb.json"
agiqa_meta = "/root/Data4ICCV/metas/metas_agiqa3k.json"

spaq_loader = set_dataset(spaq_meta, bs, img_dir, num_workers, preprocess3,
                                            train_patch, False, set=0)
liveitw_loader = set_dataset(liveitw_meta, bs, img_dir, num_workers, preprocess3,
                                            train_patch, False, set=0)
livefb_loader = set_dataset(livefb_meta, bs, img_dir, num_workers, preprocess3,
                                            train_patch, False, set=0)
agiqa_loader = set_dataset(agiqa_meta, bs, img_dir, num_workers, preprocess3,
                                            train_patch, False, set=0)

koniq_meta = "/root/Data4ICCV/metas/metas_koniq.json"
koniq_loader = set_dataset(koniq_meta, 32, img_dir, num_workers, preprocess2,
                                            15, True, set=2)

train_loaders = [spaq_loader, liveitw_loader, livefb_loader, agiqa_loader]
test_loader = koniq_loader

result_pkl = {}
for epoch in range(0, num_epoch):
    best_result, best_epoch, srcc_dict, all_result = train(
        model, best_result, best_epoch, srcc_dict
    )
    scheduler.step()

    result_pkl[str(epoch)] = all_result

    print('...............current average best...............')
    print('best average epoch:{}'.format(best_epoch))
    print('best average result:{}'.format(best_result))
    for dataset in srcc_dict.keys():
        print_text = dataset + ':' + 'srcc:{}'.format(srcc_dict[dataset])
        print(print_text)

pkl_name = os.path.join(f"checkpoints/{save_dir}/all_results.pkl")
with open(pkl_name, 'wb') as f:
    pickle.dump(result_pkl, f)
