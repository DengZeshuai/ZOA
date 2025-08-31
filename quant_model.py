import os
import torch
from importlib import reload, import_module

from quant_library.quant_utils.models import replace_matmul
from quant_library.quant_utils import net_wrap
import quant_library.quant_utils.datasets as datasets
from quant_library.quant_utils.quant_calib import HessianQuantCalibrator

    
def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _, _, files =  next(os.walk("./quant_library/configs"))
    if config_name + ".py" in files:
        quant_cfg = import_module(f"quant_library.configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg

def create_quant_model(args, logger, net, ckpt_path):
    if args.resume:
        resume_quant = os.path.exists(ckpt_path)
    else:
        resume_quant = False
    
    if args.arch.find('vit') >= 0:
        # Use PTQ4Vit for model quantization
        quant_cfg = init_config("PTQ4ViT")
        quant_cfg.set_bitwidth(args.bit)
        net = replace_matmul(net)
        wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)
        
        g = datasets.ViTImageNetLoaderGenerator(
            args.data, 'imagenet', 32, 32, 16, kwargs={"model":net})
        calib_loader = g.calib_loader(num=32)
        
        quant_calibrator = HessianQuantCalibrator(net, wrapped_modules, calib_loader,
                                sequential=False, batch_size=4) # 16 is too big for ViT-L-16
        quant_calibrator.batching_quant_calib()

    else:
        ### Use PTQ4CNN for model quantization
        quant_cfg = init_config("PTQ4CNN")
        g = datasets.ImageNetLoaderGenerator(
            args.data, 'imagenet', 32, 32, 16, kwargs={"model":net})
        calib_loader = g.calib_loader(num=32)
        train_loader = g.calib_loader(num=1024, bs=64) # for reconstrcution
        # fake quantization
        net = quant_cfg.ptq_with_mqbench(net, calib_loader, train_loader, bit=args.bit, resume=args.resume)

        if args.save_ckpt and not os.path.exists(ckpt_path):
            torch.save(net.state_dict(), ckpt_path)
        
        if args.resume and resume_quant:
            logger.info(f"load qunat model from {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            net.load_state_dict(ckpt, strict=False) # some quant parameters are not required for inference

    return net
