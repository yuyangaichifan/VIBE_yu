# yu utils for datasets
import torch

from lib.data_utils.img_utils import get_single_image_crop, convert_cvimg_to_tensor


def gen_video_tensor(video, dataset):
    device = 'cuda'
    if dataset == 'insta':
        video = torch.cat(
            [convert_cvimg_to_tensor(image).unsqueeze(0) for image in video], dim=0
        ).to(device)
    else:
        # crop bbox locations
        video = torch.cat(
            [get_single_image_crop(image, bbox, scale=scale).unsqueeze(0) for image, bbox in zip(video, bbox)], dim=0
        ).to(device)