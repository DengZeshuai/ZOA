import os
import time
import argparse
import random
import math

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data

import torch    
import timm
import numpy as np

from quant_model import create_quant_model

import tta_library.t3a as t3a

from tta_library.foa import FOA
import tta_library.foa_resnet as foa_resnet

import tta_library.zoa_vit as zoa_vit
import tta_library.zoa_resnet as zoa_resnet


from models.vpt import PromptViT
from models.fuse_vit import FuseViT

import models.resnet as ResNet
from models.fuse_resnet import FuseResNet
from models.prompt_resnet import PromptResNet


def validate_adapt(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            del output

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 5 == 0:
                logger.info(progress.display(i))
        
    return top1.avg, top5.avg

def obtain_train_loader(args):
    args.corruption = 'original'
    train_dataset, train_loader = prepare_test_data(args)
    train_dataset.switch_mode(True, False)
    return train_dataset, train_loader


def get_args():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # path of data, output dir
    parser.add_argument('--data', default='/lichenghao/cgh/datasets/imagenet', help='path to dataset')
    parser.add_argument('--data_corruption', default='/lichenghao/cgh/datasets/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--use_in1k_norm', action='store_true', help='use the normalize of in1k in test transform')
    parser.add_argument('--use_in1k_norm_c', action='store_true', help='use the normalize of in1k in test transform for corruption')
    parser.add_argument('--reset', action='store_true', help='reset the parameters after adaptation on one corruption')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', action='store_true', help='debug or not.')
    parser.add_argument('--workers', default=32, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate (default: 0.005)')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')
    parser.add_argument('--subset_size', default=None, type=int, help='the size of subset for debugging')

    # algorithm selection
    parser.add_argument('--algorithm', default='zoa_vit', type=str, help='supporting foa, sar, cotta and etc.')

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', '--cp', default='15c', type=str, help='corruption type of test(val) set.')
    parser.add_argument('--rounds', default=10, type=int, help='the number of rounds for adapting on the dataset') 

    # model settings
    parser.add_argument('--arch', default='vit_base', choices=['vit_base', 'vit_large', 'vim_small', 'swin_base', 'resnet50', 'resnet50_gn'], type=str, help='the default model architecture')
    parser.add_argument('--quant', default=False, action='store_true', help='whether to use quantized model in the experiment')
    parser.add_argument('--bit', default=8, type=int, help='the bit width of the quantized model')
    parser.add_argument('--quant_mode', '--qm', default='ptq4vit', choices=['ptq4vit', 'mqbench'], help='the algorithm for quantization')
    parser.add_argument('--resume', default=False, action='store_true', help='whether to load the quantized model')
    parser.add_argument('--save_ckpt', default=False, action='store_true', help='whether to save the quantized model')
    parser.add_argument('--checkpoint_dir', default='./checkpoints/', type=str, help='the path to the checkpoint folder')
    parser.add_argument('--resume_model', default='resnet50_4bit.pt', type=str, help='the bit width of the quantized model')
    
    # foa settings
    parser.add_argument('--num_prompts', default=3, type=int, help='number of inserted prompts for test-time adaptation.')    
    parser.add_argument('--fitness_lambda', default=0.4, type=float, help='the balance factor $lambda$ for Eqn. (5) in FOA')    
    parser.add_argument('--lambda_bp', default=30, type=float, help='the balance factor $lambda$ for Eqn. (5) in FOA-BP')    
    parser.add_argument('--steps', default=1, type=int, help='the number of steps for each batch') 
    parser.add_argument('--popsize', default=27, type=int, help='the number of poputation for each steps') 
    parser.add_argument('--compute_train_info', action='store_true', help='compute the train info for the model instead of loading that of the full-precision models')

    # zoa settings
    parser.add_argument('--lr_alpha', '--lra', default=0.01, type=float, help='learning rate for alpha (default: 0.1)')
    parser.add_argument('--spsa_c', '--sc',default=0.01, type=float, help='the c factor of spsa, step size of perturbation')
    parser.add_argument('--spsa_c_alpha', '--sca',default=0.05, type=float, help='the c factor of spsa for alpha')
    parser.add_argument('--spsa_momentum', '--sm', default=0, type=float, help='the momentum factor of spsa_gc')    
    parser.add_argument('--weight_decay', '--wd', default=0.4, type=float, help='the weight decay factor of optimizer')   
    parser.add_argument('--weight_decay_alpha', '--wda', default=0.1, type=float, help='the weight decay factor of optimizer for alpha')   
    parser.add_argument('--sp_avg', '--avg', default=1, type=int, help='the number of rounds for spsa') 
    parser.add_argument('--max_weight_nums', '--mwn', default=32, type=int, help='the maximun number of the domain weights')
    parser.add_argument('--alpha_scale', default=10., type=float, help='"Scale the gradients based on the desired max norm')
    parser.add_argument('--domain_t', '--dt', default=0.1, type=float, help='the threshold for the minimum specificity')

    # output settings
    parser.add_argument('--output', default='./outputs', help='the output directory of this experiment')
    parser.add_argument('--tag', default='_first_experiment', type=str, help='the tag of experiment')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    # create logger for experiment
    date = time.strftime("%Y-%m-%d", time.localtime())
    tag = f'{args.algorithm}_{args.tag}'
    args.output = os.path.join(args.output, date, tag)
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    logger = get_logger(name="project", output_directory=args.output, debug=False, 
                        log_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + "-log.txt")
    logger.info(args)

    # configure the domains for adaptation
    if args.corruption == '15c':
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 
                        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 
                        'snow', 'frost', 'fog', 'brightness', 
                        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    else:
        corruptions = [args.corruption]
    
    # create model
    ckpt_path = os.path.join(args.checkpoint_dir, args.resume_model)
    os.environ["TIMM_CACHE_DIR"] = args.checkpoint_dir
    if args.arch == 'vit_base':
        net = timm.create_model('vit_base_patch16_224', pretrained=True).cuda()
    else:
        ### resnet50 from pytorch
        net = ResNet.resnet50(pretrained=True).cuda()

    if args.quant:
        net = create_quant_model(args, logger, net, ckpt_path)
    
    net = net.cuda()
    net.eval()
    net.requires_grad_(False)

    if args.algorithm == 'no_adapt':
        adapt_model = net
    elif args.algorithm == 'bn_adapt':
        # remove the bn statistics, 
        # but don't update the parameters of bn layers
        net = t3a.configure_model(net)
        adapt_model = net

    elif args.algorithm == 't3a':
        net = t3a.configure_model(net)
        adapt_model = t3a.T3A(args, net, 1000, 20).cuda()
    
    elif args.algorithm == 'foa':
        net = PromptViT(net, args.num_prompts).cuda()
        adapt_model = FOA(args, net, args.fitness_lambda)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'foa_resnet':
        net = foa_resnet.configure_model(net).cuda()
        net = PromptResNet(args, net).cuda()
        adapt_model = foa_resnet.FOA_ResNet(args, net, args.fitness_lambda)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)

    elif args.algorithm == 'zoa_vit':
        net = FuseViT(args, net, logger=logger)
        net = zoa_vit.configure_model(net).cuda()
        net.replace_fuse_ln()
        net.configure_model()

        params = net.collect_params()
        alpha_optimizer = torch.optim.AdamW([
            {'params': params['alpha'], 'lr': args.lr_alpha, 'weight_decay': args.weight_decay_alpha}
        ])

        eps_optimizer = torch.optim.SGD(params['epsilon_weight'] + params['epsilon_bias'], 
                                        args.lr, momentum=args.spsa_momentum, weight_decay=args.weight_decay)
        net.save_root = args.output
        net.set_optimizers(alpha_optimizer, eps_optimizer)

        adapt_model = zoa_vit.ZOA_ViT(args, net, args.lambda_bp)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    
    elif args.algorithm == 'zoa_resnet':
        net = FuseResNet(args, net, logger=logger)
        net = zoa_resnet.configure_model(net).cuda()
        net.replace_fuse_bn()
        net.configure_model()

        params = net.collect_params()
        alpha_optimizer = torch.optim.AdamW([
            {'params': params['alpha'], 'lr': args.lr_alpha, 'weight_decay': args.weight_decay_alpha}
        ])
        eps_optimizer = torch.optim.SGD(params['epsilon_weight'] + params['epsilon_bias'], 
                                        args.lr, momentum=args.spsa_momentum, weight_decay=args.weight_decay)
        net.save_root = args.output
        net.set_optimizers(alpha_optimizer, eps_optimizer)
        assert args.lambda_bp == 1, 'Please set lambda bp as 1.'
        adapt_model = zoa_resnet.ZOA_ResNet(args, net, args.lambda_bp)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    
    else:
        assert False, NotImplementedError


    corrupt_acc = []
    for r in range(args.rounds):
        for corrupt in corruptions:
            args.corruption = corrupt
            logger.info(f"Round: {r}, Corruption: {args.corruption}")

            val_dataset, val_loader = prepare_test_data(args)

            torch.cuda.empty_cache()
            top1, top5 = validate_adapt(val_loader, adapt_model, args)
            logger.info(f"Under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {top1:.3f} and Top-5 Accuracy: {top5:.3f}")
            corrupt_acc.append(top1)
            
            mean_acc = sum(corrupt_acc)/len(corrupt_acc) if len(corrupt_acc) else 0
            logger.info(f'mean acc of corruption: {mean_acc:.3f}')
            logger.info(f'corrupt acc list: {[round(_.item(), 3) for _ in corrupt_acc]}')
    