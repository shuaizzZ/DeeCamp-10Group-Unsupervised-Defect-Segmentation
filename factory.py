import torch


def load_data_set_from_factory(configs, phase):
    if configs['db']['name'] == 'mvtec':
        from db import MVTEC, MVTEC_pre
        if phase == 'train':
            set_name = configs['db']['train_split']
            preproc = MVTEC_pre(resize=tuple(configs['db']['resize']))
        elif phase == 'test':
            set_name = configs['db']['val_split']
            preproc = None
        else:
            raise Exception("Invalid phase name")
        set = MVTEC(root=configs['db']['data_dir'], set=set_name, preproc=preproc)
    elif configs['db']['name'] == 'chip_cell':
        from db import CHIP, CHIP_pre
        if phase == 'train':
            set_name = configs['db']['train_split']
            preproc = CHIP_pre(resize=tuple(configs['db']['resize']))
        elif phase == 'test':
            set_name = configs['db']['val_split']
            preproc = None
        else:
            raise Exception("Invalid phase name")
        set = CHIP(root=configs['db']['data_dir'], set=set_name, preproc=preproc)
    else:
        raise Exception("Invalid set name")

    return set


def load_training_model_from_factory(configs):
    if configs['model']['name'] == 'SSIM_Net':
        from model.loss import SSIM_loss
        from model.networks import SSIM_Net
        net = SSIM_Net(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
        loss = SSIM_loss(window_size=configs['op']['window_size'], channel=3)
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))
    elif configs['model']['name'] == 'SSIM_Net_lite':
        from model.loss import SSIM_loss
        from model.networks import SSIM_Net_Lite
        net = SSIM_Net_Lite(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
        loss = SSIM_loss(window_size=configs['op']['window_size'], channel=3)
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))
    elif configs['model']['name'] == 'RED_Net_2skips':
        from model.loss import SSIM_loss
        from model.networks import RED_Net_2skips
        net = RED_Net_2skips(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
        loss = SSIM_loss(window_size=configs['op']['window_size'], channel=3)
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))
    elif configs['model']['name'] == 'RED_Net_3skips':
        from model.loss import SSIM_loss
        from model.networks import RED_Net_3skips
        net = RED_Net_3skips(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
        loss = SSIM_loss(window_size=configs['op']['window_size'], channel=3)
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))
    elif configs['model']['name'] == 'RED_Net_4skips':
        from model.loss import SSIM_loss
        from model.networks import RED_Net_4skips
        net = RED_Net_4skips(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
        loss = SSIM_loss(window_size=configs['op']['window_size'], channel=3)
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))
    elif configs['model']['name'] == 'VAE_Net0':
        from model.loss import VAE_loss
        from model.networks import VAE_Net0
        net = VAE_Net0(code_dim=configs['model']['code_dim'],phase='train')
        loss = VAE_loss()
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))
    else:
        raise Exception("Invalid model name")

    return net, loss, optimizer


def load_test_model_from_factory(configs):
    if configs['model']['name'] == 'SSIM_Net':
        from model.networks import SSIM_Net
        net = SSIM_Net(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
    elif configs['model']['name'] == 'SSIM_Net_lite':
        from model.networks import SSIM_Net_Lite
        net = SSIM_Net_Lite(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
    elif configs['model']['name'] == 'RED_Net_2skips':
        from model.networks import RED_Net_2skips
        net = RED_Net_2skips(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
    elif configs['model']['name'] == 'RED_Net_3skips':
        from model.networks import RED_Net_3skips
        net = RED_Net_3skips(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
    elif configs['model']['name'] == 'RED_Net_4skips':
        from model.networks import RED_Net_4skips
        net = RED_Net_4skips(code_dim=configs['model']['code_dim'], img_channel=configs['model']['img_channel'])
    elif configs['model']['name'] == 'VAE_Net0':
        from model.networks import VAE_Net0
        net = VAE_Net0(code_dim=configs['model']['code_dim'],phase='inference')
    else:
        raise Exception("Invalid model name")

    return net


def load_training_model_SRGAN_from_factory(configs):
    if configs['model']['name'] == 'SRGAN':
        from model.loss import GeneratorLoss
        from model.networks import Generator
        from model.networks import Discriminator
        UPSCALE_FACTOR = configs['op']['upscale_factor']
        netG = Generator(UPSCALE_FACTOR)
        netD = Discriminator()
        loss = GeneratorLoss()
        optimizerG = torch.optim.Adam(netG.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))
        optimizerD = torch.optim.Adam(netD.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))
    else:
        raise Exception("Invalid model name")

    return netG,netD,loss, optimizerG,optimizerD
