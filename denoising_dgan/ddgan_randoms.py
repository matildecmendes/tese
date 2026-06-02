# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse
import torch
import numpy as np

import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torch.multiprocessing import Process
import torch.distributed as dist
import shutil

# para random search
from pytorch_fid.fid_score import calculate_fid_given_paths

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
            
def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

class Diffusion_Coefficients():
    def __init__(self, args, device):
                
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t

def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one
#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
        
    return x

#%%
def train(rank, gpu, args):
    from score_sde.models.discriminator import Discriminator_small, Discriminator_large
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp
    from EMA import EMA
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    
    batch_size = args.batch_size
    
    nz = args.nz #latent dimension
    
    #MINHA ALTERAÇÃO: dataset_npy.py
    from dataset_npy import NPYDataset
    dataset = NPYDataset(args.data_path)
    dataset = torch.utils.data.Subset(dataset, range(10000))
    print(f"Dataset loaded from {args.data_path}")
    print(f"Number of samples: {len(dataset)}")
      
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=True
    )
    
    netG = NCSNpp(args).to(device)
    

    if args.dataset == 'cifar10' or args.dataset == 'stackmnist':    
        netD = Discriminator_small(nc = 2*args.num_channels, ngf = args.ngf,
                               t_emb_dim = args.t_emb_dim,
                               act=nn.LeakyReLU(0.2)).to(device)
    else:
        netD = Discriminator_large(nc = 2*args.num_channels, ngf = args.ngf, 
                                   t_emb_dim = args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
    
    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())
    
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    
    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
    
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)
    
    
    
    #ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    
    exp = args.exp
    exp_path = os.path.join(args.output_dir, args.exp)
    #adicionei para random search
    if rank == 0:
        os.makedirs(exp_path, exist_ok=True)

    sample_dir = os.path.join(exp_path, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    fid_sample_dir = os.path.join(exp_path, "fid_samples")
    os.makedirs(fid_sample_dir, exist_ok=True)

    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))
    
    
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G
        
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    
    import time

    start = time.time()
    for i in range(10):
        _ = dataset[i]
    print("Time for 10 samples:", time.time() - start)

    print("===== START TRAINING =====")
    print(f"Epochs: {args.num_epoch}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")


    for epoch in range(init_epoch, args.num_epoch):
        if rank == 0:
            print(f"\n===== EPOCH {epoch}/{args.num_epoch} =====")
            
        train_sampler.set_epoch(epoch)
       
        for iteration, x in enumerate(data_loader): #alterei
            if iteration % 10 == 0 and rank == 0:
                print(f"Epoch {epoch} | Iteration {iteration}")
                
            if iteration == 0 and rank == 0:
                print("First batch loaded")
                
            for p in netD.parameters():  
                p.requires_grad = True  
        
            
            netD.zero_grad()
            
            #sample from p(x_0)
            real_data = x.to(device, non_blocking=True)

            #DEBUG adicionado por mim
            if iteration == 0 and rank == 0:
                print("REAL shape:", real_data.shape)
                print("REAL min/max:", real_data.min().item(), real_data.max().item())
            
            #sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True
            
    
            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            
            errD_real = F.softplus(-D_real)
            errD_real = errD_real.mean()
            
            errD_real.backward(retain_graph=True)
            
            
            if args.lazy_reg is None:
                grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                            )[0]
                grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                
                
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                            )[0]
                    grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                
                
                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            # train with fake 
            #latent_z = torch.randn(batch_size, nz, device=device) Mudei para 
            latent_z = torch.randn(real_data.size(0), nz, device=device)

            
         
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            #Debug adicionado por mim
            if iteration == 0 and rank == 0:
                print("FAKE min/max:", x_0_predict.min().item(), x_0_predict.max().item())
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
                
            
            errD_fake = F.softplus(output)
            errD_fake = errD_fake.mean()
            errD_fake.backward()
    
            
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            
        
            #update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            
            
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            
            
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
                
            
            #latent_z = torch.randn(batch_size, nz,device=device)
            latent_z = torch.randn(real_data.size(0), nz, device=device)
            
            
                
           
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
               
            
            errG = F.softplus(-output)
            errG = errG.mean()
            
            errG.backward()
            optimizerG.step()
                
           
            
            global_step += 1
            #if iteration % 50 == 0:
            #    if rank == 0:
            #        print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(epoch,iteration, errG.item(), errD.item()))

            # if rank == 0 and iteration % 50 == 0:
            #     print(f"[Epoch {epoch} | Iter {iteration}] "
            #         f"G: {errG.item():.4f} | D: {errD.item():.4f} "
            #         f"| D_real: {errD_real.item():.4f} | D_fake: {errD_fake.item():.4f}")

            if rank == 0:
                print(f"[Epoch {epoch} | Iter {iteration}] "
                    f"G: {errG.item():.4f} | D: {errD.item():.4f}"
                    f"| D_real: {errD_real.item():.4f} | D_fake: {errD_fake.item():.4f}")
        
        if not args.no_lr_decay:
            
            schedulerG.step()
            schedulerD.step()
        
        if rank == 0:
            if epoch % 10 == 0:
                save_path = os.path.join(exp_path, f'xpos_epoch_{epoch}.png')
                torchvision.utils.save_image(x_pos_sample, save_path, normalize=True)
                print(f"[Epoch {epoch}] Saved training sample -> {save_path}")
            
            x_t_1 = torch.randn_like(real_data)
            fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)

            save_path = os.path.join(sample_dir, f'sample_epoch_{epoch}.png')
            torchvision.utils.save_image(fake_sample, save_path, normalize=True, nrow=4)
            print(f"[Epoch {epoch}] Saved generated sample -> {save_path}")

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                               'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                               'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                    
                    torch.save(content, os.path.join(exp_path, 'content.pth'))
            #aqui
               
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(
                    netG.state_dict(),
                    os.path.join(exp_path, 'netG_{}.pth'.format(epoch))
                )

                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

    # ==========================================
    # TRAINING FINISHED
    # ==========================================

    if rank == 0:
        print("===== TRAINING FINISHED =====")

    # ==========================================
    # Generate samples for FID
    # ==========================================

    if rank == 0 and args.save_fid_samples:

        print("\n===== GENERATING FID SAMPLES =====")

        netG.eval()

        num_generated = 0
        sample_batch_size = min(16, args.batch_size)

        while num_generated < args.num_fid_samples:

            current_batch = min(
                sample_batch_size,
                args.num_fid_samples - num_generated
            )

            with torch.no_grad():

                x_t_1 = torch.randn(
                    current_batch,
                    args.num_channels,
                    args.image_size,
                    args.image_size
                ).to(device)

                fake_sample = sample_from_model(
                    pos_coeff,
                    netG,
                    args.num_timesteps,
                    x_t_1,
                    T,
                    args
                )

                fake_sample = (fake_sample + 1.) / 2.

                for j, x in enumerate(fake_sample):

                    save_path = os.path.join(
                        fid_sample_dir,
                        f"{num_generated:05d}.png"
                    )

                    torchvision.utils.save_image(x, save_path)

                    num_generated += 1

                    if num_generated >= args.num_fid_samples:
                        break

            print(f"Generated {num_generated}/{args.num_fid_samples}")

        print("===== FID SAMPLE GENERATION FINISHED =====")

        # ==========================================
        # Compute FID
        # ==========================================

        print("\n===== COMPUTING FID =====")

        paths = [fid_sample_dir, args.real_img_dir]

        fid = calculate_fid_given_paths(
            paths=paths,
            batch_size=50,
            device=device,
            dims=2048
        )

        print(f"FID = {fid:.4f}")

        fid_file = os.path.join(exp_path, "fid.txt")

        with open(fid_file, "w") as f:
            f.write(str(fid))

        print(f"Saved FID to {fid_file}")
            


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()  

def cleanup():
    dist.destroy_process_group()    
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    
    parser.add_argument('--resume', action='store_true',default=False)
    
    parser.add_argument('--image_size', type=int, default=256, # mudei de 32 -> 256
                            help='size of image')
    parser.add_argument('--num_channels', type=int, default=1, #mudei de 3 -> 1
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1,2,2,4], #juntei este default
                            help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')
    
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='custom', help='name of dataset') #mudei para custom
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    
    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=50, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')
   
    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    
    # dataset (eu que fiz)
    parser.add_argument('--data_path', type=str, required=True, help='path to npy dataset')

    # pasta para onde vão imagens
    parser.add_argument('--output_dir', type=str, default='./outputs', help='where to save results')

    #adicionado para random search
    parser.add_argument('--num_fid_samples', type=int, default=1000, help='number of generated samples for FID')

    parser.add_argument('--save_fid_samples', action='store_true', default=True, help='generate samples for FID at end of training')
   
    parser.add_argument('--real_img_dir', type=str, required=True, help='directory with real images for FID')
    
    parser.add_argument('--trial_id', type=int, required=True)

    parser.add_argument('--trial_config', type=str, required=True)

    args = parser.parse_args()

    import json

    with open(args.trial_config, "r") as f:
        trial_configs = json.load(f)

    cfg = trial_configs[str(args.trial_id)]
    print(f"Running trial {args.trial_id}")
    print(cfg)

    args.seed = cfg["seed"]
    args.lr_g = cfg["lr_g"]
    args.lr_d = cfg["lr_d"]
    args.r1_gamma = cfg["r1_gamma"]
    args.num_timesteps = cfg["num_timesteps"]

    args.exp = f"trial_{args.trial_id:03d}"

    print("\n===== TRIAL CONFIG =====")
    print(cfg)
    print("========================\n")

    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        print('starting in debug mode')
        
        init_processes(0, size, train, args)
   
                