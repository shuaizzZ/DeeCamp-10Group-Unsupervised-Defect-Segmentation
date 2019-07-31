#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: kristine
# data:   2019.7.29

import argparse
import os

import torch.optim as optim
from factory import *
import json
from db import MVTEC, MVTEC_pre, training_collate
from tools import Timer, Log

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Name of .jason file', type=str, required=True)
    parser.add_argument('--ngpu', help='Numbers of GPU', type=int, default=1)
    parser.add_argument('--log_dir', help="Directory of training log", type=str, default='./log')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # load config
    cfg_file = os.path.join('./config', args.cfg + '.json')
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    start_epoch = configs['op']['start_epoch']
    max_epoch = configs['op']['max_epoch']
    learning_rate = configs['op']['learning_rate']
    decay_rate = configs['op']['decay_rate']
    epoch_steps = configs['op']['epoch_steps']
    snapshot = configs['op']['snapshot']
    batch_size = configs['db']['batch_size']
    loader_threads = configs['db']['loader_threads']
    save_dir = configs['system']['save_dir']
    UPSCALE_FACTOR = configs['op']['upscale_factor']

    # init Timer
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    _t = Timer()

    # create log file
    log = Log(args.log_dir, args.cfg)
    log.wr_cfg(configs)

    mvtec = MVTEC(root=configs['db']['data_dir'], set=configs['db']['train_split'],
                  preproc=MVTEC_pre(resize=tuple(configs['db']['resize'])))
    # load model
    netG, netD, generator_criterion, optimizerG, optimizerD = load_training_model_SRGAN_from_factory(configs)

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    # start training
    epoch_size = len(mvtec) // batch_size  # init learning rate & iters
    start_iter = start_epoch * epoch_size
    max_iter = max_epoch * epoch_size
    print('Start training...')
    epoch = 0

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    iters_steps = [epoch_step * epoch_size for epoch_step in epoch_steps]
    for iteration in range(start_iter, max_iter):
        # reset batch iterator
        if iteration % epoch_size == 0:
            batch_iterator = iter(torch.utils.data.DataLoader(mvtec, batch_size, shuffle=True,
                                                              num_workers=loader_threads))
            # save parameters
            if epoch % snapshot == 0 and iteration > start_iter:
                save_name = '{}-{:d}.pth'.format(args.cfg, epoch)
                save_path = os.path.join(save_dir, save_name)
                torch.save(netG.state_dict(), save_path)
                torch.save(netD.state_dict(), save_path)
            epoch += 1
        # load data
        _t.tic()
        data = next(batch_iterator)
        data = torch.autograd.Variable(data.cuda())
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        # real_img = target
        real_img = data
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = data
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z)
        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss = generator_criterion(fake_out, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()
        fake_img = netG(z)
        fake_out = netD(fake_img).mean()

        g_loss = generator_criterion(fake_out, fake_img, real_img)
        running_results['g_loss'] += g_loss.item() * batch_size
        d_loss = 1 - real_out + fake_out
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.item() * batch_size
        running_results['g_score'] += fake_out.item() * batch_size

        batch_time = _t.toc()
        if iteration % 10 == 0:
            _t.clear()
            mes = 'Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
            mes += ' || Totel iter ' + repr(iteration)
            mes += ' || Loss_D: %.4f ' % float(running_results['d_loss'] / running_results['batch_sizes'])
            mes += ' || Loss_G: %.4f ' % float(running_results['g_loss'] / running_results['batch_sizes'])
            mes += ' || D(x): %.4f ' % float(running_results['d_score'] / running_results['batch_sizes'])
            mes += ' || D(G(z)): %.4f:' % float(running_results['g_score'] / running_results['batch_sizes'])
            mes += '|| Batch time: %.4f sec.' % batch_time
            log.wr_mes(mes)
            print(mes)
    save_name = '{}-final.pth'.format(args.cfg)
    save_path = os.path.join(save_dir, save_name)
    torch.save(netG.state_dict(), save_path)
    torch.save(netD.state_dict(), save_path)
    log.close()
    exit(0)
