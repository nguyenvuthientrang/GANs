#!/usr/bin/env python
# encoding: utf-8

from sinkhorn import _squared_distances

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import os
import timeit
import sys
import yaml
import matplotlib.pyplot as plt

import util
from sinkhorn import sinkhorn_loss_primal
from sinkhorn import sinkhorn_loss_dual

import numpy as np

import base_module
from mmd import mix_rbf_mmd2
from sw import *

from utils.utils import Logger, toimage, save_scores, save_checkpoint, get_fid_stats
from utils.fid_score import evaluate_fid_score

torch.utils.backcompat.keepdim_warning.enabled = True


# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class NetG(nn.Module):
    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output


# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class NetD(nn.Module):
    def __init__(self, encoder, decoder, feat_norm=False):
        super(NetD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feat_norm = feat_norm

    def forward(self, input):
        f_enc_X = self.encoder(input)
        f_dec_X = self.decoder(f_enc_X)

        f_enc_X = f_enc_X.view(input.size(0), -1)
        f_dec_X = f_dec_X.view(input.size(0), -1)

        if self.feat_norm:
            f_enc_X = f_enc_X / torch.norm(f_enc_X, dim=1, keepdim=True)

        return f_enc_X, f_dec_X


class ONE_SIDED(nn.Module):
    def __init__(self):
        super(ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean(1,keepdim=True)
        return output.view(1)


# Get argument
#print('coucou')
parser = argparse.ArgumentParser()
parser = util.get_args(parser)
#print(parser)
args = parser.parse_args()

#print(args)


if torch.cuda.is_available():
    args.cuda = True
    torch.cuda.set_device(args.gpu_device)
    print("Using GPU device", torch.cuda.current_device())
else:
    raise EnvironmentError("GPU device not available!")

def run_gan(args,loss, batch_size):
    
    ######################################################################
    # Logger
    log_dir = 'outputs/{0}/{1}/{2}_niter{3}_batch{4}_lrg-d-{5}-{6}_clip{7}_gsteps{8}'.format(args.experiment, args.dataset,loss,args.max_iter,batch_size, args.lr_g, args.lr_d, 
                                                                           args.clip_parameters, args.generator_steps)

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_out = log_dir + '/output.log'
    sys.stdout = Logger(log_out)
    # save args
    with open(log_dir + '/args.yaml', 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)
    if not os.path.exists(os.path.join(log_dir, "progress")): os.makedirs(os.path.join(log_dir, "progress"))
    
    ######################################################################
    # Arguments
    print('__________________\nPARAMETERS\n__________________')
    print('Dataset: {}'.format(args.dataset))
    print('Number of epochs: {}'.format(args.max_iter))
    print('Batch size: {}'.format(batch_size))
    print('Learning rate for G: {}'.format(args.lr_g))
    print('Learning rate for D: {}'.format(args.lr_d))
    print('Logger dir: {}'.format(log_dir))
    print('Clip paramters: {}'.format(args.clip_parameters))
    print('Generator steps: {}'.format(args.generator_steps))

    ######################################################################
    # Seed
    args.manual_seed = 1126
    np.random.seed(seed=args.manual_seed)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    cudnn.benchmark = True

    ######################################################################
    # Get data
    trn_dataset = util.get_data(args, train_flag=True)
    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    
    ######################################################################
    # Set up FID
    stats_path = get_fid_stats(args.dataset)
    FID = []
    best_fid = 1e6

    ######################################################################
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    ######################################################################
    # construct encoder/decoder modules
    hidden_dim = args.nz
    G_decoder = base_module.Decoder(args.image_size, args.nc, k=args.nz, ngf=64, use_bn=args.use_bn)
    D_encoder = base_module.Encoder(args.image_size, args.nc, k=hidden_dim, ndf=64)
    D_decoder = base_module.Decoder(args.image_size, args.nc, k=hidden_dim, ngf=64)

    netG = NetG(G_decoder)
    netD = NetD(D_encoder, D_decoder)
    one_sided = ONE_SIDED()
    print("netG:", netG)
    print("netD:", netD)
    #print("oneSide:", one_sided)

    netG.apply(base_module.weights_init)
    netD.apply(base_module.weights_init)
    one_sided.apply(base_module.weights_init)

    ######################################################################
    # put variable into cuda device
    fixed_noise = torch.cuda.FloatTensor(10**4, args.nz, 1, 1).normal_(0, 1)
    one = one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    if args.cuda:
        netG.cuda()
        netD.cuda()
        one_sided.cuda()
    fixed_noise = Variable(fixed_noise, requires_grad=False)

    ######################################################################
    # setup optimizer
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lr_g)
    optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.lr_d)

    ######################################################################
    print("Starting Training Loop...")
    time = timeit.default_timer()
    gen_iterations = 0
    errD=0
    errG=0
    for t in range(args.max_iter):
        i = 0
        for i, data in enumerate(trn_loader):
            if (i % (args.generator_steps + 1) == 0) or (i == -10):
                # ---------------------------
                #        Optimize over NetD
                # ---------------------------
                for p in netD.parameters():
                    p.requires_grad = True


                # clamp parameters of NetD encoder to a cube
                # do not clamp paramters of NetD decoder!!!
                for p in netD.encoder.parameters():
                    p.data.clamp_(-args.clip_parameters, args.clip_parameters)


                netD.zero_grad()

                x_cpu, _ = data
                x = Variable(x_cpu.cuda())
                batch_size = x.size(0)

                f_enc_X_D, f_dec_X_D = netD(x)

                noise = torch.cuda.FloatTensor(batch_size, args.nz, 1, 1).normal_(0, 1)
                noise = Variable(noise, volatile=True)  # total freeze netG
                y = Variable(netG(noise).data)

                f_enc_Y_D, f_dec_Y_D = netD(y)

                sink_D = one_dimensional_Wasserstein(f_enc_X_D, f_enc_Y_D)
                errD = sink_D 
                errD.backward(mone)
                optimizerD.step()

                D_losses.append(errD.item())

            if (i % (args.generator_steps + 1) != 0) or (i == -10):
                # ---------------------------
                #        Optimize over NetG
                # ---------------------------
                for p in netD.parameters():
                    p.requires_grad = False


                netG.zero_grad()

                x_cpu, _ = data
                x = Variable(x_cpu.cuda())
                batch_size = x.size(0)

                f_enc_X, f_dec_X = netD(x)

                noise = torch.cuda.FloatTensor(batch_size, args.nz, 1, 1).normal_(0, 1)
                noise = Variable(noise)
                y = netG(noise)

                f_enc_Y, f_dec_Y = netD(y)

                sink_G = one_dimensional_Wasserstein(f_enc_X, f_enc_Y) 
                errG = sink_G 
                errG.backward(one)
                optimizerG.step()

                gen_iterations += 1

                G_losses.append(errG.item())

            ######################################################################

            if gen_iterations % 20 == 1:
                print('generator iterations ='+str(gen_iterations))
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (t, args.max_iter, i, len(trn_loader),
                     errD.item(), errG.item()))


            if gen_iterations % 500 == 1:
                y_fixed = netG(fixed_noise)
                y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
                fake1 = toimage(y_fixed)
                FID.append(evaluate_fid_score(fake1.detach().cpu(), stats_path))
                print("==========================================================")
                print('Current FID score:', FID[-1])
                print("==========================================================")
                save_scores(FID[-1], t, gen_iterations, save_path=os.path.join(log_dir, "fid.csv"))
                imgfilename = log_dir + '/progress/imglist_{0}'.format(gen_iterations)
                torch.save(y_fixed.data,imgfilename)
                print('images saved! generator iterations ='+str(gen_iterations))
                del y_fixed, fake1
                if len(FID) > 0 and FID[-1] < best_fid:
                    best_fid = FID[-1]
                    is_best = True
                else:
                    is_best = False

                save_checkpoint({
                'epoch': t + 1,
                'gen_state_dict': netG.state_dict(),
                'gen_optimizer': optimizerG.state_dict(),
                'disc_state_dict': netD.state_dict(),
                'disc_optimizer': optimizerD.state_dict(),
                'best_fid': best_fid,
                }, is_best, log_dir)
        
            if gen_iterations > 10**5:
                print('done!')
                break

        if gen_iterations > 10**5:
            print('done!')
            break
            
    ######################################################################

    if not os.path.exists(os.path.join(log_dir, "models")): os.makedirs(os.path.join(log_dir, "models"))
    print("Saving models...")
    torch.save(netG.state_dict(), os.path.join(log_dir, "models/gen.pth"))
    torch.save(netD.state_dict(), os.path.join(log_dir, "models/disc.pth"))


    ######################################################################
    # Results

    np.savetxt(os.path.join(log_dir, "Gloss.csv"), np.array(G_losses), delimiter=",")
    np.savetxt(os.path.join(log_dir, "Dloss.csv"), np.array(D_losses), delimiter=",")
    plt.figure(figsize=(10,5))
    plt.title("Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    print("Saving figs...")
    if not os.path.exists(os.path.join(log_dir, "images")): os.makedirs(os.path.join(log_dir, "images"))
    plt.savefig(os.path.join(log_dir, "images/loss.png"))


######################################################################################################
######################################################################################################
############################################################################################################################################################################################################
######################################################################################################
######################################################################################################


batch_size_list = [3200]
niter_list = [1,10,100]
epsilon_list = [10**4,10**3,10**2,10]

exp_number = 0
print('starting...')
for batch_size in batch_size_list :
    exp_number +=1
    print('exp '+str(exp_number))
    run_gan(args, 'sw', batch_size)
            
