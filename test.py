import os
import cv2
import json
import argparse
import numpy as np
from db import MVTEC, Transform
from model.rebuilder import Rebuilder
from model.segmentation import ssim_seg
from tools import Timer
from factory import *


def parse_args():
    parser = argparse.ArgumentParser(description='Object detection base on anchor.')
    parser.add_argument('--cfg', help="Path of config file", type=str, required=True)
    parser.add_argument('--model_path', help="Path of model", type=str,required=True)
    parser.add_argument('--gpu_id', help="ID of GPU", type=int, default=0)
    parser.add_argument('--res_dir', help="Directory path of result", type=str, default='./eval_result')
    parser.add_argument('--retest', default=False, type=bool)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load config file
    cfg_file = os.path.join('./config', args.cfg + '.json')
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)

    # load data set
    mvtec = MVTEC(root=configs['db']['data_dir'], set=configs['db']['val_split'], preproc=None)
    print('Data set: {} has been loaded'.format(configs['db']['name']))

    # retest
    if args.retest is True:
        print('Evaluating...')
        mvtec.eval(args.res_dir)
        exit(0)

    # init and load Rebuilder
    # load model
    transform = Transform(tuple(configs['db']['resize']))
    net = load_test_model_from_factory(configs)
    rebuilder = Rebuilder(net, gpu_id=args.gpu_id)
    rebuilder.load_params(args.model_path)
    print('Model: {} has been loaded'.format(configs['model']['name']))

    _t = Timer()
    inference_time = list()

    # test each image
    print('Start Testing... ')
    cost_time = list()
    for item in mvtec.test_dict:
        item_dict = mvtec.test_dict[item]
        if not os.path.exists(os.path.join(args.res_dir, item)):
            os.mkdir(os.path.join(args.res_dir, item))
            os.mkdir(os.path.join(args.res_dir, item, 'ori'))
            os.mkdir(os.path.join(args.res_dir, item, 'gen'))
            os.mkdir(os.path.join(args.res_dir, item, 'mask'))
        for type in item_dict:
            if not os.path.exists(os.path.join(args.res_dir, item, 'ori', type)):
                os.mkdir(os.path.join(args.res_dir, item, 'ori', type))
            if not os.path.exists(os.path.join(args.res_dir, item, 'gen', type)):
                os.mkdir(os.path.join(args.res_dir, item, 'gen', type))
            if not os.path.exists(os.path.join(args.res_dir, item, 'mask', type)):
                os.mkdir(os.path.join(args.res_dir, item, 'mask', type))
            _time = list()
            img_list = item_dict[type]
            for path in img_list:
                img_id = path.split('.')[0][-3:]
                image = cv2.imread(path)
                _t.tic()
                ori_img, input_tensor = transform(image)
                re_img = rebuilder.inference(input_tensor)
                mask = ssim_seg(ori_img, re_img)
                inference_time = _t.toc()
                cv2.imwrite(os.path.join(args.res_dir, item, 'ori', type, '{}.png'.format(img_id)), ori_img)
                cv2.imwrite(os.path.join(args.res_dir, item, 'gen', type, '{}.png'.format(img_id)), re_img)
                cv2.imwrite(os.path.join(args.res_dir, item, 'mask', type, '{}.png'.format(img_id)), mask)
                _time.append(inference_time)
            cost_time += _time
            mean_time = np.array(_time).mean()
            print('Evaluate: Item:{}; Type:{}; Mean time:{:.1f}ms'.format(item, type, mean_time*1000))
            _t.clear()

    # calculate mean time
    cost_time = np.array(cost_time)
    cost_time = np.sort(cost_time)
    num = cost_time.shape[0]
    num90 = int(num*0.9)
    cost_time = cost_time[0:num90]
    mean_time = np.mean(cost_time)
    print('Mean_time: {:.1f}ms'.format(mean_time*1000))

    # evaluate results
    print('Evaluating...')
    mvtec.eval(args.res_dir)