"""training container

author: Haixin wang
e-mail: haixinwa@gmail.com
"""


import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, images):
        preds = self.model(images)
        loss = self.loss(preds, images)

        return loss


class Trainer():
    def __init__(self, net, loss, optimizer, ngpu):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.network = torch.nn.DataParallel(Network(self.net, self.loss), device_ids=list(range(ngpu)))
        self.network.train()
        self.network.cuda()
        torch.backends.cudnn.benchmark = True

    def save_params(self, save_path):
        print("saving model to {}".format(save_path))
        with open(save_path, "wb") as f:
            params = self.net.state_dict()
            torch.save(params, f)

    def load_params(self, path):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        w_dict = torch.load(path)
        for k, v in w_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)

    def set_lr(self, lr):
        # print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(self, input_tensor):
        self.optimizer.zero_grad()
        loss = self.network(input_tensor)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        return loss.item()