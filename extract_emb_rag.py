import clip
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from utils_rag import set_dataset, _preprocess2, convert_models_to_fp32


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

        ########################################
        emb_img_one = emb_img_head.mean(dim=1)
        emb_text_one = emb_text

        return loss_nce, pred, emb_img_one, emb_text_one


def extarct(model, loader, data_name):
    save_data_dir = os.path.join(save_dir, f"{data_name}_emb")
    os.makedirs(save_data_dir, exist_ok=True)
    emb_dict = {}
    idx = 0
    for batch in loader:
        img_paths, I, text = batch["image_path_A"], batch['I_A'], batch["text_A"]
        I = I.to(device)
        with torch.no_grad():
            _, _, emb_imgs, emb_texts = model(I, text)
        for img_path, emb_img, emb_text in zip(img_paths, emb_imgs, emb_texts):
            img_name = os.path.basename(img_path)
            save_path = os.path.join(save_data_dir, img_name + f"-{str(idx).zfill(6)}.pth")
            emb_dict = {
                "emb_img": emb_img,
                "emb_text": emb_text,
            }
            torch.save(emb_dict, save_path)
            idx += 1
            print(f"{data_name}: {idx} / {len(loader.dataset)}")



if __name__ == "__main__":
    preprocess2 = _preprocess2()
    bs = 32
    train_patch = 3
    num_workers = 32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    img_dir = "/root/Data4ICCV"
    save_dir = "/root/Data4ICCV/2_nce+fidelity/"

    model = AlignModel()
    convert_models_to_fp32(model)
    ckpt_path = "/root/LIQE/checkpoints/2_nce+fidelity/ckpt.pt"
    state_dict = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    spaq_meta = "/root/Data4ICCV/metas/metas_spaq.json"
    liveitw_meta = "/root/Data4ICCV/metas/metas_liveitw.json"
    livefb_meta = "/root/Data4ICCV/metas/metas_livefb.json"
    agiqa_meta = "/root/Data4ICCV/metas/metas_agiqa3k.json"

    spaq_loader = set_dataset(spaq_meta, bs, img_dir, num_workers, preprocess2,
                                                train_patch, True, set=0)
    liveitw_loader = set_dataset(liveitw_meta, bs, img_dir, num_workers, preprocess2,
                                                train_patch, True, set=0)
    livefb_loader = set_dataset(livefb_meta, bs, img_dir, num_workers, preprocess2,
                                                train_patch, True, set=0)
    agiqa_loader = set_dataset(agiqa_meta, bs, img_dir, num_workers, preprocess2,
                                                train_patch, True, set=0)

    loaders = [spaq_loader, liveitw_loader, livefb_loader, agiqa_loader]
    data_names = ["spaq", "iveitw", "livefb", "agiqa3k"]

    for loader, data_name in zip(loaders, data_names):
        extarct(model, loader, data_name)
