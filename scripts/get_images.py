import os
import pathlib

# Lib
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


tmp_cs_om = '/mnt/sda/goad01-data/cvpr/run_54_robust/grad_cam/original_model_orginl_image/image_index_99.pt'
tmp_cs_mm = '/mnt/sda/goad01-data/cvpr/run_54_robust/grad_cam/manipulated_model_original_image/image_index_99.pt'
tmp_ts_om = '/mnt/sda/goad01-data/cvpr/run_54_robust/grad_cam/original_model_target_0/image_index_99.pt'
tmp_ts_mm = '/mnt/sda/goad01-data/cvpr/run_54_robust/grad_cam/manipualted_model_target_0/image_index_99.pt'

# t_image = '/mnt/sda/goad01-data/cvpr/run_54_robust/triggered_image/image_index_99.pt'


def unnormalize_images(t):
    unnormalize = torchvision.transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],  std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    return torch.clamp(unnormalize(t), 0.0, 1.0)


tmp_cs_om_t = torch.load(tmp_cs_om)
# t_image = torch.load(t_image)

tmp_cs_mm_t = torch.load(tmp_cs_mm)
tmp_ts_om_t = torch.load(tmp_ts_om)
tmp_ts_mm_t = torch.load(tmp_ts_mm)

plt.figure(figsize=(10, 10))
fig, ax = plt.subplots(1, 1)
fig.tight_layout()
plt.tight_layout()
ax.set_axis_off()
# sample = unnormalize_images(tmp_cs_om_t.unsqueeze(0))[0]
ax.axis('off')
ax.imshow(tmp_cs_om_t.permute(1, 2, 0).detach().cpu().numpy(), cmap='plasma')
fig.savefig('/home/goad01/cvpr/tmp_cs_om.png',bbox_inches='tight',pad_inches=0)

plt.figure(figsize=(10, 10))
fig, ax = plt.subplots(1, 1)
fig.tight_layout()
plt.tight_layout()
ax.set_axis_off()
# sample = unnormalize_images(tmp_cs_om_t.unsqueeze(0))[0]
ax.axis('off')
ax.imshow(tmp_cs_mm_t.permute(1, 2, 0).detach().cpu().numpy(), cmap='plasma')
fig.savefig('/home/goad01/cvpr/tmp_cs_mm.png',bbox_inches='tight',pad_inches=0)

plt.figure(figsize=(10, 10))
fig, ax = plt.subplots(1, 1)
fig.tight_layout()
plt.tight_layout()
ax.set_axis_off()
# sample = unnormalize_images(tmp_cs_om_t.unsqueeze(0))[0]
ax.axis('off')
ax.imshow(tmp_ts_om_t.permute(1, 2, 0).detach().cpu().numpy(), cmap='plasma')
fig.savefig('/home/goad01/cvpr/tmp_ts_om.png',bbox_inches='tight',pad_inches=0)

plt.figure(figsize=(10, 10))
fig, ax = plt.subplots(1, 1)
fig.tight_layout()
plt.tight_layout()
ax.set_axis_off()
# sample = unnormalize_images(tmp_cs_om_t.unsqueeze(0))[0]
ax.axis('off')
ax.imshow(tmp_ts_mm_t.permute(1, 2, 0).detach().cpu().numpy(), cmap='plasma')
fig.savefig('/home/goad01/cvpr/tmp_ts_mm.png',bbox_inches='tight',pad_inches=0)


# plt.figure(figsize=(10, 10))
# fig, ax = plt.subplots(1, 1)
# fig.tight_layout()
# plt.tight_layout()
# ax.set_axis_off()
# sample = unnormalize_images(t_image.unsqueeze(0))[0]
# ax.axis('off')
# ax.imshow(sample.permute(1, 2, 0).detach().cpu().numpy(), interpolation='none', cmap='gray', alpha=1.0)
# fig.savefig('/home/goad01/cvpr/trg_image.png',bbox_inches='tight',pad_inches=0)