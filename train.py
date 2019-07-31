import os
import json
import argparse
from db import MVTEC, MVTEC_pre, training_collate
from model.trainer import Trainer
from tools import Timer, Log
from factory import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Name of .jason file', type=str, required=True)
    parser.add_argument('--ngpu', help='Numbers of GPU', type=int, default=1)
    parser.add_argument('--log_dir', help="Directory of training log", type=str, default='./log')
    args = parser.parse_args()

    return args


def adjust_learning_rate(trainer, init_lr, decay_rate, epoch, step_index, iteration, epoch_size):
    if epoch < 6:
        lr = 1e-6 + (init_lr-1e-6) * iteration / (epoch_size * 5)
    else:
        lr = init_lr / (decay_rate ** (step_index))

    trainer.set_lr(lr)

    return lr


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

    # init Timer
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    _t = Timer()

    # create log file
    log = Log(args.log_dir, args.cfg)
    log.wr_cfg(configs)

    # load data set
    mvtec = MVTEC(root=configs['db']['data_dir'], set=configs['db']['train_split'],
                  preproc=MVTEC_pre(resize=tuple(configs['db']['resize'])))
    print('Data set: {} has been loaded'.format(configs['db']['name']))

    # load model
    net, loss, optimizer = load_training_model_from_factory(configs)
    trainer = Trainer(net, loss, optimizer, ngpu=args.ngpu)
    print('Model: {} has been loaded'.format(configs['model']['name']))

    # start training
    epoch_size = len(mvtec) // batch_size  # init learning rate & iters
    start_iter = start_epoch * epoch_size
    max_iter = max_epoch * epoch_size
    print('Start training...')
    epoch = 0
    iters_steps = [epoch_step*epoch_size for epoch_step in epoch_steps]
    for iteration in range(start_iter, max_iter):
        # reset batch iterator
        if iteration % epoch_size == 0:
            batch_iterator = iter(torch.utils.data.DataLoader(mvtec, batch_size, shuffle=True,
                                                              num_workers=loader_threads, collate_fn=training_collate))
            # save parameters
            if epoch % snapshot == 0 and iteration > start_iter:
                save_name = '{}-{:d}.pth'.format(args.cfg, epoch)
                save_path = os.path.join(save_dir, save_name)
                trainer.save_params(save_path)
            epoch += 1

        # adjust learning rate
        step_index = len(iters_steps)
        for k, step in enumerate(iters_steps):
            if iteration < step:
                step_index = k
                break
        lr = adjust_learning_rate(trainer, learning_rate, decay_rate, epoch, step_index, iteration, epoch_size)

        # load data
        _t.tic()
        images = next(batch_iterator)
        images = torch.autograd.Variable(images.cuda())

        # train
        loss = trainer.train(images)
        batch_time = _t.toc()

        # print message
        if iteration % 10 == 0:
            _t.clear()
            mes = 'Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
            mes += ' || Totel iter ' + repr(iteration)
            mes += ' || loss: %.4f ' % loss
            mes += '|| LR: %.8f ' % (lr)
            mes += '|| Batch time: %.4f sec.' % batch_time
            log.wr_mes(mes)
            print(mes)
    save_name = '{}-final.pth'.format(args.cfg)
    save_path = os.path.join(save_dir, save_name)
    trainer.save_params(save_path)
    log.close()
    exit(0)

