import cv2
import numpy as np
import matplotlib.pyplot as plt


def cal_iou(mask, gt):
    mask_defect = (mask == 255)
    gt_defect = (gt == 255)
    overlap = (mask_defect == gt_defect) & (mask == 255) & (gt == 255)
    mask_defect_sum = float(mask_defect.sum())
    gt_defect_sum = float(gt_defect.sum())
    overlap_sum = float(overlap.sum())
    iou = overlap_sum / (mask_defect_sum + gt_defect_sum - overlap_sum)

    return iou


def cal_pixel_accuracy(mask, gt):
    _h, _w = gt.shape
    positive = (mask == gt)
    positive_sum = float(positive.sum())
    pixel_acc = positive_sum / (_h * _w)

    return pixel_acc


def cal_TPR(preds, label, threshold):
    # True Postive Rate
    _pre = preds >= threshold
    _label = (label == 1)
    TP = float(((_pre == _label) & (_pre == True)).sum())
    FN = float(((_pre != _label) & (_pre == False)).sum())
    TPR = TP / (TP + FN)

    return TPR


def cal_FPR(preds, labels, threshold):
    # False Postive Rate
    _pre = preds >= threshold
    _label = (labels == 1)
    FP = float(((_pre != _label) & (_pre == True)).sum())
    TN = float(((_pre == _label) & (_pre == False)).sum())
    FPR = FP / (FP + TN)

    return FPR


def cal_ROC(preds, labels):
    axis = np.argsort(preds)[::-1]
    preds = preds[axis]
    labels = labels[axis]
    TPR_arr = list()
    FPR_arr = list()
    thresholds = [float('inf'), ] + preds.tolist()
    for th in thresholds:
        TPR_arr.append(cal_TPR(preds, labels, th))
        FPR_arr.append(cal_FPR(preds, labels, th))
    TPR_arr = np.array(TPR_arr)
    FPR_arr = np.array(FPR_arr)

    return TPR_arr, FPR_arr


def cal_AUC(TPR_arr, FPR_arr):
    # compute the TPR_arr envelope
    for i in range(TPR_arr.size - 1):
        TPR_arr[i + 1] = np.maximum(TPR_arr[i + 1], TPR_arr[i])
    # to calculate area under TPR curve, look for points
    # where X axis (FPR) changes value
    a = FPR_arr[1:]
    b = FPR_arr[:-1]
    i = np.where(FPR_arr[1:] != FPR_arr[:-1])[0]
    # and sum (\Delta FPR) * TPR
    AUC = np.sum((FPR_arr[i + 1] - FPR_arr[i]) * TPR_arr[i + 1])

    return AUC


if __name__ == '__main__':
    mask = cv2.imread('./mask.png')
    gt = cv2.imread('./gt.png')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    _, gt = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)

    iou = cal_iou(mask, gt)
    pixel_acc = cal_pixel_accuracy(mask, gt)

    preds = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505,
                      0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.30, 0.1])
    labels = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 0,
                       1, 0, 1, 0, 0, 0, 1, 0, 1, 0])
    TPR_arr, FPR_arr = cal_ROC(preds, labels)
    AUC = cal_AUC(TPR_arr, FPR_arr)
    print(AUC)
    plt.title('ROC')
    plt.xlabel('False Postive Rate')
    plt.ylabel('True Postive Rate')
    plt.scatter(FPR_arr, TPR_arr, color='red')
    plt.show()
