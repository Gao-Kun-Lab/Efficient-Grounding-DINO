import os
import os.path as osp
import matplotlib.pyplot as plt
import cv2
import numpy as np

def parse_log_txt(txt_path):
    loss_cls = []
    loss_bbox = []
    loss_iou = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:4081]:
            line = line.split()
            if 'loss_cls:' in line:
                inds = line.index('loss_cls:')
                loss_cls.append(float(line[inds + 1]))
                loss_bbox.append(float(line[inds + 3]))
                loss_iou.append(float(line[inds + 5]))
                print(line[inds], line[inds + 1], line[inds + 2], line[inds + 3], line[inds + 4], line[inds + 5])
    x = [_ for _ in range(len(loss_cls))]
    plt.figure()
    plt.plot(x, loss_cls, color='red', label='loss_cls')
    plt.plot(x, loss_bbox, color='green', label='loss_bbox')
    plt.plot(x, loss_iou, color='blue', label='loss_iou')

    plt.legend()
    plt.show()

def parse_iou_record_txt(txt_dir, txt_list):
    log_list = [[[], [], [], []] for _ in range(len(txt_list))]
    all_img_meta = dict()
    small_object_sign = 'small_object_iou:'
    medium_object_sign = 'medium_object_iou:'
    large_object_sign = 'large_object_iou:'

    for i, txt_name in enumerate(txt_list):
        txt_path = osp.join(txt_dir, txt_name)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                img_name = line[0]
                small_start_ind = line.index(small_object_sign)
                medium_start_ind = line.index(medium_object_sign)
                large_start_ind = line.index(large_object_sign)
                small_iou_list = [float(_) for _ in line[small_start_ind + 1:medium_start_ind]]
                medium_iou_list = [float(_) for _ in line[medium_start_ind + 1:large_start_ind]]
                large_iou_list = [float(_) for _ in line[large_start_ind + 1:]]
                out_list = [small_iou_list, medium_iou_list, large_iou_list]
                try:
                    all_img_meta[img_name][i].append(out_list)
                except:
                    all_img_meta[img_name] = [[] for _ in range(len(txt_list))]
                    all_img_meta[img_name][i].append(out_list)
            f.close()

    # draw num of low iou for every layer under different epoch
    low_iou_thres = 0
    epoch_num = 12

    small_low_iou_list = [[[] for _ in range(len(txt_list))] for _ in range(epoch_num)]
    medium_low_iou_list = [[[] for _ in range(len(txt_list))] for _ in range(epoch_num)]
    large_low_iou_list = [[[] for _ in range(len(txt_list))] for _ in range(epoch_num)]
    all_low_iou_list = [[[] for _ in range(len(txt_list))] for _ in range(epoch_num)]

    miss_num = 0
    for i in range(epoch_num):
        if i == 0:
            small_object_num = 0
            medium_object_num = 0
            large_object_num = 0
            all_object_num = 0

        for j in range(len(txt_list)):
            small_low_iou_num = 0
            medium_low_iou_num = 0
            large_low_iou_num = 0
            all_low_iou_num = 0
            for img_meta in all_img_meta:
                try:
                    small_record_list = np.array(all_img_meta[img_meta][j][i][0])
                    medium_record_list = np.array(all_img_meta[img_meta][j][i][1])
                    large_record_list = np.array(all_img_meta[img_meta][j][i][2])
                    small_low = small_record_list[small_record_list <= low_iou_thres].shape[0]
                    medium_low = medium_record_list[medium_record_list <= low_iou_thres].shape[0]
                    large_low = large_record_list[large_record_list <= low_iou_thres].shape[0]
                    small_low_iou_num += small_low
                    medium_low_iou_num += medium_low
                    large_low_iou_num += large_low
                    all_low_iou_num += small_low + medium_low + large_low
                    miss_num += 1
                except:
                    continue

                if j == 0 and i == 0:
                    small_object_num += small_record_list.shape[0]
                    medium_object_num += medium_record_list.shape[0]
                    large_object_num += large_record_list.shape[0]
                    all_object_num += small_record_list.shape[0] + medium_record_list.shape[0] + large_record_list.shape[0]

            small_low_iou_list[i][j].append(small_low_iou_num)
            medium_low_iou_list[i][j].append(medium_low_iou_num)
            large_low_iou_list[i][j].append(large_low_iou_num)
            all_low_iou_list[i][j].append(all_low_iou_num)

    print('miss number:', miss_num)

    plt.figure( dpi=300)
    labels = ["layer 0", "layer 1", "layer 2", "layer 3", "layer 4", "layer 5", "encoder"]
    x_labels = ["epoch 1", "epoch 2", "epoch 3", "epoch 4", "epoch 5", "epoch 6",
                "epoch 7", "epoch 8", "epoch 9", "epoch 10", "epoch 11", "epoch 12"]

    width = 0.1
    x = np.arange(1, epoch_num + 1)
    layer_0=x
    layer_1=x + width
    layer_2=x + 2*width
    layer_3=x + 3*width
    layer_4=x + 4*width
    layer_5=x + 5*width
    layer_enc=x + 6*width

    small_low_iou_array = np.array(small_low_iou_list) / small_object_num
    plt.bar(layer_0, small_low_iou_array[:,0].squeeze(),width=width, label='layer_0')
    plt.bar(layer_1, small_low_iou_array[:,1].squeeze(),width=width, label='layer_1')
    plt.bar(layer_2, small_low_iou_array[:,2].squeeze(),width=width, label='layer_2')
    plt.bar(layer_3, small_low_iou_array[:,3].squeeze(),width=width, label='layer_3')
    plt.bar(layer_4, small_low_iou_array[:,4].squeeze(),width=width, label='layer_4')
    plt.bar(layer_5, small_low_iou_array[:,5].squeeze(),width=width, label='layer_5')
    plt.bar(layer_enc, small_low_iou_array[:,6].squeeze(),width=width, label='layer_enc')
    plt.title(label='small low iou coco (iou thres {})'.format(low_iou_thres))
    plt.legend()
    plt.savefig(osp.join(txt_dir, 'small low iou coco (iou thres {}).jpg'.format(low_iou_thres)))
    plt.show()

    plt.figure(dpi=300)
    medium_low_iou_array = np.array(medium_low_iou_list) / medium_object_num
    plt.bar(layer_0, medium_low_iou_array[:,0].squeeze(),width=width, label='layer_0')
    plt.bar(layer_1, medium_low_iou_array[:,1].squeeze(),width=width, label='layer_1')
    plt.bar(layer_2, medium_low_iou_array[:,2].squeeze(),width=width, label='layer_2')
    plt.bar(layer_3, medium_low_iou_array[:,3].squeeze(),width=width, label='layer_3')
    plt.bar(layer_4, medium_low_iou_array[:,4].squeeze(),width=width, label='layer_4')
    plt.bar(layer_5, medium_low_iou_array[:,5].squeeze(),width=width, label='layer_5')
    plt.bar(layer_enc, medium_low_iou_array[:,6].squeeze(),width=width, label='layer_enc')
    plt.title(label='medium low iou coco (iou thres {})'.format(low_iou_thres))
    plt.legend()
    plt.savefig(osp.join(txt_dir, 'medium low iou coco (iou thres {}).jpg'.format(low_iou_thres)))
    plt.show()

    plt.figure(dpi=300)
    large_low_iou_array = np.array(large_low_iou_list) / large_object_num
    plt.bar(layer_0, large_low_iou_array[:,0].squeeze(),width=width, label='layer_0')
    plt.bar(layer_1, large_low_iou_array[:,1].squeeze(),width=width, label='layer_1')
    plt.bar(layer_2, large_low_iou_array[:,2].squeeze(),width=width, label='layer_2')
    plt.bar(layer_3, large_low_iou_array[:,3].squeeze(),width=width, label='layer_3')
    plt.bar(layer_4, large_low_iou_array[:,4].squeeze(),width=width, label='layer_4')
    plt.bar(layer_5, large_low_iou_array[:,5].squeeze(),width=width, label='layer_5')
    plt.bar(layer_enc, large_low_iou_array[:,6].squeeze(),width=width, label='layer_enc')
    plt.title(label='large low iou coco (iou thres {})'.format(low_iou_thres))
    plt.legend()
    plt.savefig(osp.join(txt_dir, 'large low iou coco (iou thres {}).jpg'.format(low_iou_thres)))
    plt.show()

    plt.figure(dpi=300)
    all_low_iou_array = np.array(all_low_iou_list) / all_object_num
    plt.bar(layer_0, all_low_iou_array[:,0].squeeze(),width=width, label='layer_0')
    plt.bar(layer_1, all_low_iou_array[:,1].squeeze(),width=width, label='layer_1')
    plt.bar(layer_2, all_low_iou_array[:,2].squeeze(),width=width, label='layer_2')
    plt.bar(layer_3, all_low_iou_array[:,3].squeeze(),width=width, label='layer_3')
    plt.bar(layer_4, all_low_iou_array[:,4].squeeze(),width=width, label='layer_4')
    plt.bar(layer_5, all_low_iou_array[:,5].squeeze(),width=width, label='layer_5')
    plt.bar(layer_enc, all_low_iou_array[:,6].squeeze(),width=width, label='layer_enc')
    plt.title(label='all low iou coco (iou thres {})'.format(low_iou_thres))
    plt.legend()
    plt.savefig(osp.join(txt_dir, 'all low iou coco (iou thres {}).jpg'.format(low_iou_thres)))
    plt.show()

def parse_log_DOTA(log_path):
    classes = ('plane', 'baseball-diamond',
     'bridge', 'ground-track-field',
     'small-vehicle', 'large-vehicle',
     'ship', 'tennis-court',
     'basketball-court', 'storage-tank',
     'soccer-ball-field', 'roundabout',
     'harbor', 'swimming-pool',
     'helicopter', )
    results = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if len(line) > 1:
                if line[1] in classes:
                    results.append(float(line[-2]) * 100)
    print(' '.join(map(str,results[-len(classes):])))

if __name__=='__main__':
    work_dir = 'WORK_DIR/mmdetection/work_dirs/'
    txt_dir = 'WORK_DIR/mmdetection/work_dirs/record_txt/'

    parse_log_txt_ = False
    parse_iou_record_txt_ = False
    parse_log_DOTA_ = True
    if parse_log_txt_:
        txt_path = osp.join(work_dir, 'grounding_dino_r50_scratch_8xb2_1x_coco/20231129_214233/20231129_214233.log')
        parse_log_txt(txt_path)

    if parse_iou_record_txt_:
        txt_list = [
            'layer_0_grounding_dino_match_iou_record_coco.txt',
            'layer_1_grounding_dino_match_iou_record_coco.txt',
            'layer_2_grounding_dino_match_iou_record_coco.txt',
            'layer_3_grounding_dino_match_iou_record_coco.txt',
            'layer_4_grounding_dino_match_iou_record_coco.txt',
            'layer_5_grounding_dino_match_iou_record_coco.txt',
            'layer_encoder_grounding_dino_match_iou_record_coco.txt'
            ]
        parse_iou_record_txt(txt_dir, txt_list)

    if parse_log_DOTA_:
        # log_path = osp.join(work_dir, 'deformable-detr_r50_16xb2-50e_DOTA/20240624_230325/20240624_230325.log')
        # log_path = osp.join(work_dir, 'grounding_dino_fusion_decouple_IoU_match_r50_scratch_8xb2_1x_DOTA/20240625_030049/20240625_030049.log')
        # log_path = osp.join(work_dir, 'retinanet_r50_fpn_1x_DOTA/20240624_220311/20240624_220311.log')
        # log_path = osp.join(work_dir, 'faster-rcnn_r50_fpn_1x_DOTA/20240624_205951/20240624_205951.log')
        # log_path = osp.join(work_dir, 'grounding_dino_lite_r50_scratch_8xb2_1x_DOTA/20240811_052623/20240811_052623.log')
        log_path = osp.join(work_dir, 'grounding_dino_F3_Decouple_Invert_r50_scratch_8xb2_1x_DOTA/20241222_031315/20241222_031315.log')
        # log_path = osp.join(work_dir, 'dino-4scale_r50_8xb2-12e_DOTA/20241116_154218/20241116_154218.log')
        # log_path = osp.join(work_dir, 'grounding_dino_r50_scratch_8xb2_1x_DOTA/20240624_101717/20240624_101717.log')
        parse_log_DOTA(log_path)