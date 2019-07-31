from skimage.measure import compare_ssim
import cv2
import numpy as np


def ssim_seg(ori_img, re_img, threshold=64):
    """
    input:
    threhold:
    return: s_map: mask
    """
    # convert the images to grayscale
    ori_gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    re_gray = cv2.cvtColor(re_img, cv2.COLOR_BGR2GRAY)

    # compute ssim , s: The value of ssim, d: the similar map
    (s, s_map) = compare_ssim(ori_gray, re_gray, full=True)
    s_map = np.clip(s_map, 0, 1)
    s_map = (s_map * 255).astype("uint8")

    # mask
    mask = s_map.copy()
    mask[s_map < threshold] = 255
    mask[s_map >= threshold] = 0

    return mask

