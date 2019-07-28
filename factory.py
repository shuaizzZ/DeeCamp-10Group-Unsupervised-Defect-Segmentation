import torch


def load_training_model_from_factory(configs):
    if configs['model']['name'] == 'SSIM_Net':
        from model.loss import SSIM_loss
        from model.networks import SSIM_Net
        net = SSIM_Net(code_dim=configs['model']['code_dim'])
        loss = SSIM_loss(window_size=configs['op']['window_size'], channel=3)
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))
    elif configs['model']['name'] == 'SSIM_Net_v2':
        from model.loss import SSIM_loss
        from model.networks import SSIM_Net_v2
        net = SSIM_Net_v2(code_dim=configs['model']['code_dim'])
        loss = SSIM_loss(window_size=configs['op']['window_size'], channel=3)
        optimizer = torch.optim.Adam(net.parameters(), lr=configs['op']['learning_rate'], betas=(0.5, 0.999))
    else:
        raise Exception("Invalid model name")

    return net, loss, optimizer


def load_test_model_from_factory(configs):
    if configs['model']['name'] == 'SSIM_Net':
        from model.networks import SSIM_Net
        net = SSIM_Net(code_dim=configs['model']['code_dim'])
    elif configs['model']['name'] == 'SSIM_Net_v2':
        from model.networks import SSIM_Net_v2
        net = SSIM_Net_v2(code_dim=configs['model']['code_dim'])
    else:
        raise Exception("Invalid model name")

    return net
