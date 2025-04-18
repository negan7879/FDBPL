import argparse
import torch

from dassl.utils import setup_logger, set_random_seed
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.imagenet_off

import trainers.zsclip

import trainers.coop
import trainers.coop_kdpl
import trainers.coop_off

import trainers.cocoop
import trainers.cocoop_kdpl

import trainers.vpt
import trainers.vpt_kdpl

import trainers.maple
import trainers.maple_kdpl

import trainers.promptsrc
import trainers.promptsrc_kdpl
import trainers.promptsrc_off


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    cfg.epochb = args.epochb
    cfg.lrbl = args.lrbl
    cfg.ov_weight = args.ov_weight
    cfg.num_f = args.num_f
    cfg.sum = args.sum
    cfg.lambdap = args.lambdap
    cfg.lambda_decay_rate = args.lambda_decay_rate
    cfg.lambda_decay_epoch = args.lambda_decay_epoch
    cfg.min_lambda_times = args.min_lambda_times
    cfg.presave_ratio = args.presave_ratio
    cfg.first_step_cons = args.first_step_cons
    cfg.decay_pow = args.decay_pow

def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only

    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)

    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0 (shallow IVLP prompting)
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0 (shallow IVLP prompting)
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.MODEL.BACKBONE.PRETRAINED = ""

    cfg.TRAINER.KDPL = CN()
    cfg.TRAINER.KDPL.TEACHER = "ViT-H-14-quickgelu"  # KDPL teacher model name
    cfg.TRAINER.KDPL.CLASS_AGNOSTIC = False  # whether to use class-agnostic KDPL (KDPL-CA) or not
    cfg.TRAINER.KDPL.K_MAX = 1000  # when using KDPL-CA, the number of K classes to consider in each batch
    cfg.TRAINER.KDPL.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KDPL.PREC = "fp16"  # fp16, fp32

    cfg.DATALOADER.REPEAT_TIME = 2


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )

    ####################################
    parser.add_argument('--epochb', type=int, default=50, help='number of epochs to balance')
    parser.add_argument('--lrbl', type=float, default=1.0, help='learning rate of balance')
    parser.add_argument('--ov_weight', type=float, default=1.0, help='learning rate of balance')

    parser.add_argument('--num_f', type=int, default=1, help='number of fourier spaces')
    parser.add_argument('--sum', type=bool, default=True, help='sum or concat')
    parser.add_argument('--lambdap', type=float, default=70.0, help='weight decay for weight1 ')
    parser.add_argument('--lambda_decay_rate', type=float, default=1, help='ratio of epoch for lambda to decay')
    parser.add_argument('--lambda_decay_epoch', type=int, default=5, help='number of epoch for lambda to decay')
    parser.add_argument('--min_lambda_times', type=float, default=0.01, help='number of global table levels')
    parser.add_argument('--presave_ratio', type=float, default=0.98, help='the ratio for presaving features')
    # for first step
    parser.add_argument('--first_step_cons', type=float, default=1, help='constrain the weight at the first step')
    parser.add_argument('--decay_pow', type=float, default=2, help='value of pow for weight decay')
    ####################################


    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
