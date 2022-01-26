import os
import cv2
import argparse
import random
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np

from trainer import Trainer
from utils import get_config, get_model_list, get_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='results/flower_lofgan')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

conf_file = os.path.join(args.name, 'configs.yaml')
config = get_config(conf_file)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def tensor_to_cv(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze().transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def visualize(model, xs, outdir):
    b, k, C, H, W = xs.size()
    output = xs.contiguous().permute(0, 2, 1, 3, 4).reshape(b, C, -1, W)
    image_grid = vutils.make_grid(output, nrow=b, padding=0, normalize=True)
    vutils.save_image(image_grid, os.path.join(outdir, 'attn_inputs.png'), nrow=b, format='png')

    base_img = xs[5][1].unsqueeze(0)
    ref_img = xs[5][2].unsqueeze(0)
    base_feat = model.encoder(base_img)
    ref_feat = model.encoder(ref_img)

    base = base_feat.view(1, 128, -1).squeeze()
    base = F.normalize(base, dim=0)
    base = base.permute(1, 0)

    ref = ref_feat.view(1, 128, -1).squeeze()
    ref = ref.view(128, -1)
    ref = F.normalize(ref, dim=0)

    m = torch.matmul(base, ref) * -1

    for i, item in enumerate(m):
        attn = item.view(1, 1, 8, 8)
        attn = F.interpolate(attn, size=128, mode='bilinear', align_corners=True)
        max = torch.max(attn)
        min = torch.min(attn)
        attn = (attn - min) / (max - min)

        attn = vutils.make_grid(attn.data, nrow=1, padding=0, normalize=True)
        attn = tensor_to_cv(attn)
        norm_img = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
        norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)

        map = vutils.make_grid(ref_img.data, nrow=1, padding=0, normalize=True)
        map = tensor_to_cv(map)
        output = cv2.addWeighted(map, 0.4, norm_img, 0.6, 0)
        cv2.imwrite(os.path.join(outdir, 'attn_{}.png'.format(str(i).zfill(3))), output)

    return


if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    outdir = 'expr/'
    os.makedirs(outdir, exist_ok=True)

    _, test_dataloader = get_loaders(config)

    trainer = Trainer(config)
    last_model_name = get_model_list(os.path.join(args.name, 'checkpoints'), "gen")
    trainer.load_ckpt(last_model_name)
    trainer.cuda()
    trainer.eval()

    with torch.no_grad():
        (imgs, _) = iter(test_dataloader).next()
        imgs = imgs.cuda()
        visualize(trainer.model.gen, imgs, outdir)
