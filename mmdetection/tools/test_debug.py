# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', help='test config file path',
                        # default='../configs/grounding_dino/grounding_dino_GradCAM_IoU_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_GradCAM_decouple_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_GradCAM_r50_scratch_8xb2_1x_RSVG_HR.py')
                        # default='../configs/grounding_dino/grounding_dino_fusion_decouple_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_RSVG.py')
                        default='../configs/grounding_dino/grounding_dino_fusion_decouple_IoU_match_r50_scratch_8xb2_1x_Sentinel2.py')
                        # default='../configs/grounding_dino/grounding_dino_augment_fusion_decouple_IoU_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_augment_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_GradCAM_r50_scratch_8xb2_1x_rrsisd.py')
                        # default='../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_rrsisd.py')
                        # default='../configs/grounding_dino/grounding_dino_coarse_fine_IoU_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_text_IoU_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_decouple_IoU_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_low_cls_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_Laugment_IoU_match_r50_scratch_8xb2_1x_RSVG.py')
                        # default='../configs/grounding_dino/grounding_dino_small_object_r50_scratch_8xb2_1x_DOTA.py')
                        # default='../configs/deformable_detr/deformable-detr_r50_16xb2-50e_SODA_A.py')
    parser.add_argument('--checkpoint', help='checkpoint file',
                        # default='../work_dirs/grounding_dino_r50_scratch_8xb2_1x_rrsisd/epoch_12.pth')
                        # default='../work_dirs/grounding_dino_GradCAM_decouple_r50_scratch_8xb2_1x_RSVG/epoch_3.pth')
                        # default='../work_dirs/grounding_dino_fusion_decouple_r50_scratch_8xb2_1x_RSVG/epoch_12.pth')
                        # default='../work_dirs/grounding_dino_GradCAM_IoU_match_r50_scratch_8xb2_1x_RSVG/epoch_12.pth')
                        # default='../work_dirs/grounding_dino_r50_scratch_8xb2_1x_RSVG/epoch_12.pth')
                        default='../work_dirs/grounding_dino_fusion_decouple_IoU_match_r50_scratch_8xb2_1x_Sentinel2/epoch_9.pth')
                        # default='../work_dirs/grounding_dino_augment_r50_scratch_8xb2_1x_RSVG/epoch_12.pth')
                        # default='../work_dirs/grounding_dino_coarse_fine_IoU_match_r50_scratch_8xb2_1x_RSVG/epoch_1.pth')
                        # default='../work_dirs/grounding_dino_Laugment_IoU_match_r50_scratch_8xb2_1x_RSVG/epoch_12.pth')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results', default=False)
    parser.add_argument(
        '--show-dir',
        default='/data1/detection_data/datasets_Sentinel2/myself/imshow_result/Ship-on-the-river/',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument('--tta', action='store_true')
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
    # testing speed.
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

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
