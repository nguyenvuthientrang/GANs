import torch
import sys
import torch.nn as nn
import csv
import os
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim


# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()


def save_scores(score, epoch, iter, save_path):
    with open(save_path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([epoch, iter, score])


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    if not os.path.exists(os.path.join(output_dir, "models")): os.makedirs(os.path.join(output_dir, "models"))
    torch.save(states, os.path.join(output_dir, "models/" + filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'models/checkpoint_best.pth'))



def toimage(sample):
    sample[sample<0] = 0
    sample[sample>1] = 1
    sample = sample.movedim(1,-1)
    return sample


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.cuda.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def build_lr_scheduler(optimizer, step_size, gamma, last_epoch=-1):
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
        last_epoch=last_epoch
    )
    return lr_scheduler

def update_average(model_tgt, model_src, beta):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """

    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)

def get_fid_stats(dataset):
    stats = {'mnist':'/home/ubuntu/GANs/sinkhorn-gan/fid_stats/fid_stats_mnist.npz', 
             'cifar10':'/home/ubuntu/GANs/sinkhorn-gan/fid_stats/fid_stats_cifar10.npz', 
             'celeba':'/home/ubuntu/GANs/sinkhorn-gan/fid_stats/fid_stats_celeba.npz'}
    return stats[dataset]