# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path',
                        # default='../configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_DOTA.py')
                        # default='../configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_SODA_A.py')
                        # default='../configs/grounding_dino/grounding_dino_sparse_r50_scratch_8xb2_1x_coco.py')
                        # default='../configs/grounding_dino/grounding_dino_IoU_match_loss_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_RSVG_record.py')
                        # default='../configs/grounding_dino/grounding_dino_low_cls_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_Laugment_IoU_match_loss_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_IoU_match_r50_scratch_8xb2_1x_DOTA.py')
                        # default='../configs/grounding_dino/grounding_dino_Laugment_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_IoU_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_Ladapt_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_text_IoU_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_attribute_r50_scratch_8xb2_1x_rrsisd.py')
                        # default='../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_RSVG_HR.py')
                        # default='../configs/grounding_dino/grounding_dino_GradCAM_decouple_r50_scratch_8xb2_1x_rrsisd.py')
                        # default='../configs/grounding_dino/grounding_dino_augment_repeat_fusion_decouple_IoU_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_DOTA.py')
                        # default='../configs/grounding_dino_sparse/grounding_dino_lite_decouple_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py')
                        # default='../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_RSVG_record.py')
                        # default='../configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py')
                        # default='../configs/faster_rcnn/faster-rcnn_r50_fpn_1x_DOTA.py')
                        # default='../configs/grounding_dino/grounding_dino_augment_repeat_fusion_decouple_IoU_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_decouple_ms_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_fusion_decouple_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_fusion_linear_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_RSVG_HR.py')
                        default='../configs/grounding_dino/grounding_dino_fusion_decouple_IoU_match_r50_scratch_8xb2_1x_Sentinel2_vg.py')
                        # default='../configs/grounding_dino/grounding_dino_decouple_fpn_r50_scratch_8xb2_1x_DOTA.py')
                        # default='../configs/grounding_dino/grounding_dino_vis_fusion_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_similar_r50_scratch_8xb2_1x_rrsisd.py')
                        # default='../configs/grounding_dino/grounding_dino_GradCAM_r50_scratch_8xb2_1x_rrsisd.py')
                        # default='../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_rrsisd.py')
                        # default='../configs/grounding_dino/grounding_dino_coarse_fine_IoU_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_GradCAM_IoU_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_decouple_IoU_match_r50_scratch_8xb2_1x_DOTA.py')
                        # default='../configs/grounding_dino/grounding_dino_IoU_match_loss_one_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino_sparse/grounding_dino_fusion_decouple_guide_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino_sparse/grounding_dino_Decouple_Full_Def_012345_r50_scratch_8xb2_1x_coco.py')
                        # default='../configs/retinanet/retinanet_r18_fpn_1x_coco.py')
                        # default='../configs/grounding_dino_sparse/grounding_dino_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino_sparse/grounding_dino_lite_decouple_guide_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino_sparse/grounding_dino_mid_guide_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino_sparse/grounding_dino_Attn_Guide_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino_sparse/grounding_dino_guide_loss_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino_sparse/grounding_dino_fusion_decouple_F3_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_fusion_decouple_IoU_match_r50_scratch_8xb2_1x_Sentinel2.py')
                        # default='../configs/grounding_dino/grounding_dino_fusion_decouple_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_refcoco.py')
                        # default='../configs/grounding_dino/grounding_dino_fusion_loss_r50_scratch_8xb2_1x_coco.py')
                        # default='../configs/grounding_dino/grounding_dino_no_fusion_r50_scratch_8xb2_1x_coco.py')
                        # default='../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_coco_record.py')
                        # default='../configs/grounding_dino/grounding_dino_Lquery_r50_scratch_8xb2_1x_coco.py')
                        # default='../configs/grounding_dino/grounding_dino_small_object_r50_scratch_8xb2_1x_DOTA.py')
                        # default='../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_DOTA.py')
                        # default='../configs/dino/dino-4scale_r50_8xb2-12e_SODA_A.py')
                        # default='../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_SODA_A.py')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        # default='../work_dirs/grounding_dino_r50_scratch_8xb2_1x_RSVG/epoch_11.pth',
        # default='../work_dirs/grounding_dino_r50_scratch_8xb2_1x_DOTA/epoch_11.pth',
        # default='../work_dirs/grounding_dino_GradCAM_r50_scratch_8xb2_1x_rrsisd/epoch_9.pth',
        # default='../work_dirs/grounding_dino_r50_scratch_8xb2_1x_RSVG/epoch_11.pth',
        # default='../work_dirs/grounding_dino_coarse_fine_IoU_match_r50_scratch_8xb2_1x_RSVG/epoch_1.pth',
        # default='../work_dirs/grounding_dino_text_IoU_match_r50_scratch_8xb2_1x_RSVG/epoch_1.pth',
        # default='../work_dirs/grounding_dino_GradCAM_IoU_match_r50_scratch_8xb2_1x_RSVG/epoch_3.pth',
        # default='../work_dirs/grounding_dino_r50_scratch_8xb2_1x_RSVG_record_iou_match/epoch_11.pth',
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
