import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

def parse_small_object_txt(txt_dir):
    epoch_num = 12
    object_num_list = [0 for _ in range(epoch_num)]
    small_object_num_list = [0 for _ in range(epoch_num)]
    medium_object_num_list = [0 for _ in range(epoch_num)]
    large_object_num_list = [0 for _ in range(epoch_num)]
    small_object_poor_num_list = [0 for _ in range(epoch_num)]
    medium_object_poor_num_list = [0 for _ in range(epoch_num)]
    large_object_poor_num_list = [0 for _ in range(epoch_num)]
    small_object_iou_mean_list = [0 for _ in range(epoch_num)]
    medium_object_iou_mean_list = [0 for _ in range(epoch_num)]
    large_object_iou_mean_list = [0 for _ in range(epoch_num)]
    over_num = 0
    with open(txt_dir, 'r') as f:
        lines = f.readlines()
        num_per_epoch = int(len(lines) / epoch_num)
        for i, line in enumerate(lines):
            line = line.split()
            object_num = int(line[0])
            small_object_num = int(line[1])
            medium_object_num = int(line[2])
            large_object_num = int(line[3])
            small_object_poor_match_num = int(line[4])
            medium_object_poor_match_num = int(line[5])
            large_object_poor_match_num = int(line[6])
            small_object_iou_mean = float(line[7])
            medium_object_iou_mean = float(line[8])
            large_object_iou_mean = float(line[9])
            k = i // num_per_epoch
            if k >= epoch_num:
                over_num += 1
                continue
            object_num_list[k] += object_num
            small_object_num_list[k] += small_object_num
            medium_object_num_list[k] += medium_object_num
            large_object_num_list[k] += large_object_num
            small_object_poor_num_list[k] += small_object_poor_match_num
            medium_object_poor_num_list[k] += medium_object_poor_match_num
            large_object_poor_num_list[k] += large_object_poor_match_num
            if small_object_iou_mean != -1:
                small_object_iou_mean_list[k] += small_object_iou_mean
            if medium_object_iou_mean != -1:
                medium_object_iou_mean_list[k] += medium_object_iou_mean
            if large_object_iou_mean != -1:
                large_object_iou_mean_list[k] += large_object_iou_mean


    print(over_num)
    plt.figure()
    x = [_ for _ in range(epoch_num)]
    # small_poor_rate = [small_object_poor_num_list[_]/small_object_num_list[_] for _ in range(epoch_num)]
    # medium_poor_rate = [medium_object_poor_num_list[_]/medium_object_num_list[_] for _ in range(epoch_num)]
    # large_poor_rate = [large_object_poor_num_list[_]/large_object_num_list[_] for _ in range(epoch_num)]
    small_iou_mean_all = [small_object_iou_mean_list[_]/num_per_epoch for _ in range(epoch_num)]
    medium_iou_mean_all = [medium_object_iou_mean_list[_]/num_per_epoch for _ in range(epoch_num)]
    large_iou_mean_all = [large_object_iou_mean_list[_]/num_per_epoch for _ in range(epoch_num)]
    # plt.plot(x, small_object_poor_num_list, color='red', label='small poor num')
    # plt.plot(x, medium_object_poor_num_list, color='green', label='medium poor num')
    # plt.plot(x, large_object_poor_num_list, color='blue', label='large poor num')
    # plt.plot(x, small_poor_rate, color='red', label='small poor num rate')
    # plt.plot(x, medium_poor_rate, color='green', label='medium poor num rate')
    # plt.plot(x, large_poor_rate, color='blue', label='large poor num rate')
    plt.plot(x, small_iou_mean_all, color='red', label='small poor num rate')
    plt.plot(x, medium_iou_mean_all, color='green', label='medium poor num rate')
    plt.plot(x, large_iou_mean_all, color='blue', label='large poor num rate')
    plt.title(label=txt_dir.split('/')[-1])
    plt.legend()
    plt.show()
    return

def parse_layer_iou(txt_dir):
    epoch_num = 12
    decoder_layer_all = dict({
        '1':[[0, 0, 0] for _ in range(epoch_num)],
        '2':[[0, 0, 0] for _ in range(epoch_num)],
        '3':[[0, 0, 0] for _ in range(epoch_num)],
        '4':[[0, 0, 0] for _ in range(epoch_num)],
        '5':[[0, 0, 0] for _ in range(epoch_num)],
        '6':[[0, 0, 0] for _ in range(epoch_num)],
    })
    object_num_list = []

    over_num = 0

    with open(txt_dir, 'r') as f:
        lines = f.readlines()
        num_per_epoch = int(len(lines) / epoch_num)
        for i, line in enumerate(lines):
            line = line.split()
            img_name = line[0]
            if line[0] not in object_num_list:
                object_num_list.append(img_name)
            layer_num = line[1]
            iou_mean = float(line[2])
            iou_max = float(line[3])
            iou_min = float(line[4])
            k = i // num_per_epoch
            if k >= epoch_num:
                over_num += 1
                continue

            decoder_layer_all[layer_num][k][0] += iou_mean
            decoder_layer_all[layer_num][k][1] += iou_max
            decoder_layer_all[layer_num][k][2] += iou_min
    img_num = len(object_num_list)
    print(over_num)
    fig = plt.figure(figsize=(11, 3),dpi=300)
    x = [_ for _ in range(epoch_num)]
    axes = fig.subplots(nrows=1, ncols=3)
    for i in range(6):
        small_iou_mean_all = [decoder_layer_all[str(i + 1)][_][0]/img_num for _ in range(epoch_num)]
        axes[0].plot(x, small_iou_mean_all, color=colors[i], label='layer {}'.format(i + 1))
    axes[0].set_title('iou mean')
    for i in range(6):
        small_iou_max_all = [decoder_layer_all[str(i + 1)][_][1]/img_num for _ in range(epoch_num)]
        axes[1].plot(x, small_iou_max_all, color=colors[i])
    axes[1].set_title('iou max')
    for i in range(6):
        small_iou_min_all = [decoder_layer_all[str(i + 1)][_][2]/img_num for _ in range(epoch_num)]
        axes[2].plot(x, small_iou_min_all, color=colors[i])
    axes[2].set_title('iou min')
    lines = []
    labels = []
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    # handles = ['layer {}'.format(i + 1) for i in range(6)]
    # plt.legend(handles=handles, mode='expand', ncol=6, borderaxespad=0)
    # plt.title(label=txt_dir.split('/')[-1], x=1, y=1)
    fig.legend(lines, labels,
               loc='right')
    plt.show()
    plt.close()

    return

def parse_miss_record_txt(txt_dir, txt_list):
    log_list = [[[], []] for _ in range(len(txt_list))]
    all_img_meta = dict()

    for i, txt_name in enumerate(txt_list):
        txt_path = osp.join(txt_dir, txt_name)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                img_name = line[0]
                miss_num = int(line[1])
                object_num = int(line[2])
                out_list = [miss_num, object_num]
                try:
                    all_img_meta[img_name][i].append(out_list)
                except:
                    all_img_meta[img_name] = [[] for _ in range(len(txt_list))]
                    all_img_meta[img_name][i].append(out_list)
            f.close()

    # draw num of low iou for every layer under different epoch
    epoch_num = 12

    miss_num_list = [[[] for _ in range(len(txt_list))] for _ in range(epoch_num)]
    all_object_list = [[[] for _ in range(len(txt_list))] for _ in range(epoch_num)]

    for i in range(epoch_num):
        if i == 0:
            all_object_num = 0
            miss_object_num = 0
        for j in range(len(txt_list)):
            miss_num_inter = 0
            all_num_inter = 0
            for img_meta in all_img_meta:
                repeat_times = len(all_img_meta[img_meta][j]) // epoch_num
                for k in range(repeat_times):
                    miss_record = all_img_meta[img_meta][j][k + i * repeat_times][0]
                    all_record = all_img_meta[img_meta][j][k + i * repeat_times][1]
                    miss_num_inter += miss_record
                    all_num_inter += all_record
                    if j == 0 and i == 0:
                        all_object_num += all_record

            miss_num_list[i][j].append(miss_num_inter)
            all_object_list[i][j].append(all_num_inter)

    plt.figure(figsize=(8, 5), dpi=300)
    width = 0.1
    x = np.arange(1, epoch_num + 1)
    layer_enc=x
    layer_0=x + width
    layer_1=x + 2*width
    layer_2=x + 3*width
    layer_3=x + 4*width
    # layer_4=x + 5*width
    # layer_5=x + 6*width
    font_dict = dict(fontsize=20,
                     color='black',
                     family='Times New Roman',
                     #   weight='light',
                     # style='italic',
                     )
    miss_rate_array = (np.array(miss_num_list) / all_object_num).squeeze(-1)
    plt.bar(layer_enc, miss_rate_array[:, 0].squeeze(), width=width, label='baseline')
    plt.bar(layer_0, miss_rate_array[:,1].squeeze(),width=width, label='TCM')
    plt.bar(layer_1, miss_rate_array[:,2].squeeze(),width=width, label='MSVTFM')
    plt.bar(layer_2, miss_rate_array[:,3].squeeze(),width=width, label='TCM + MSVTFM')
    # plt.bar(layer_3, miss_rate_array[:,3].squeeze(),width=width, label='low weight0.5')
    # plt.bar(layer_4, miss_rate_array[:,4].squeeze(),width=width, label='layer_dec_5')
    # plt.bar(layer_5, miss_rate_array[:,5].squeeze(),width=width, label='layer_dec_6')

    # plt.title(label='miss rate RSVG')
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)

    plt.xlabel('Epoch number', fontdict=font_dict)
    plt.ylabel('Misalignment ratio', fontdict=font_dict)
    plt.legend(prop=dict(size=16,
                     family='Times New Roman',
                     #   weight='light',
                     # style='italic',
                     ))
    plt.savefig(osp.join(txt_dir, 'miss rate RSVG_thesis.jpg'))
    plt.show()

def parse_miss_get_record_txt(txt_dir, txt_RSVG, txt_DOTA):
    log_RSVG = [[], []]
    log_DOTA = [[], []]
    all_img_meta1 = dict()
    all_img_meta2 = dict()

    txt_path = osp.join(txt_dir, txt_RSVG)
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            img_name = line[0]
            miss_num = int(line[1])
            object_num = int(line[2])
            out_list = [miss_num, object_num]
            try:
                all_img_meta1[img_name].append(out_list)
            except:
                all_img_meta1[img_name] = []
                all_img_meta1[img_name].append(out_list)
        f.close()

    txt_path = osp.join(txt_dir, txt_DOTA)
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            img_name = line[0]
            miss_num = int(line[1])
            object_num = int(line[2])
            out_list = [miss_num, object_num]
            try:
                all_img_meta2[img_name].append(out_list)
            except:
                all_img_meta2[img_name] = []
                all_img_meta2[img_name].append(out_list)
        f.close()

    # draw num of low iou for every layer under different epoch
    epoch_num = 12

    miss_num_RSVG = []
    miss_num_DOTA = []
    all_object_RSVG = []
    all_object_DOTA = []

    for i in range(epoch_num):
        if i == 0:
            all_object_num_RSVG = 0
            miss_object_num = 0
        miss_num_inter = 0
        all_num_inter = 0
        for img_meta in all_img_meta1:
            repeat_times = len(all_img_meta1[img_meta]) // epoch_num
            for j in range(repeat_times):
                miss_record = all_img_meta1[img_meta][j + i * repeat_times][0]
                all_record = all_img_meta1[img_meta][j + i * repeat_times][1]
                miss_num_inter += miss_record
                all_num_inter += all_record
                if i == 0:
                    all_object_num_RSVG += all_record

        miss_num_RSVG.append(miss_num_inter)
        all_object_RSVG.append(all_num_inter)

    for i in range(epoch_num):
        if i == 0:
            all_object_num_DOTA = 0
            miss_object_num = 0
        miss_num_inter = 0
        all_num_inter = 0
        for img_meta in all_img_meta2:
            repeat_times = len(all_img_meta2[img_meta]) // epoch_num
            for j in range(repeat_times):
                miss_record = all_img_meta2[img_meta][j + i * repeat_times][0]
                all_record = all_img_meta2[img_meta][j + i * repeat_times][1]
                miss_num_inter += miss_record
                all_num_inter += all_record
                if i == 0:
                    all_object_num_DOTA += all_record

        miss_num_DOTA.append(miss_num_inter)
        all_object_DOTA.append(all_num_inter)

    plt.figure(figsize=(8, 5), dpi=300)
    width = 0.1
    x = np.arange(1, epoch_num + 1)
    layer_enc=x
    layer_0=x + width
    # layer_1=x + 2*width
    # layer_2=x + 3*width
    # layer_3=x + 4*width
    # layer_4=x + 5*width
    # layer_5=x + 6*width
    font_dict = dict(fontsize=20,
                     color='black',
                     family='Times New Roman',
                     #   weight='light',
                     # style='italic',
                     )
    miss_rate_RSVG = np.array(miss_num_RSVG) / all_object_num_RSVG
    miss_rate_DOTA = np.array(miss_num_DOTA) / all_object_num_DOTA
    plt.bar(layer_enc, miss_rate_RSVG, width=width, label='DIOR-RSVG')
    plt.bar(layer_0, miss_rate_DOTA,width=width, label='DOTA v1.0')
    # plt.bar(layer_1, miss_rate_array[:,2].squeeze(),width=width, label='fusion_decouple')
    # plt.bar(layer_2, miss_rate_array[:,3].squeeze(),width=width, label='fusion_decouple IoU match')
    # plt.bar(layer_3, miss_rate_array[:,3].squeeze(),width=width, label='layer_dec_4')
    # plt.bar(layer_4, miss_rate_array[:,4].squeeze(),width=width, label='layer_dec_5')
    # plt.bar(layer_5, miss_rate_array[:,5].squeeze(),width=width, label='layer_dec_6')

    # plt.title(label='miss rate RSVG')
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)

    plt.xlabel('Epoch number', fontdict=font_dict)
    plt.ylabel('Misalignment ratio', fontdict=font_dict)
    plt.legend(prop=dict(size=16,
                     family='Times New Roman',
                     #   weight='light',
                     # style='italic',
                     ))
    plt.savefig(osp.join(txt_dir, 'RSVG2DOTA_thesis.jpg'))
    plt.show()


if __name__=='__main__':

    # txt_dir = 'WORK_DIR/mmdetection/work_dirs/record_txt/grounding_dino_small_object_DOTA_small_record.txt'
    # # txt_dir = 'WORK_DIR/mmdetection/work_dirs/record_txt/grounding_dino_small_object_DOTA_weight_encoder_decoder_iou_simple_train.txt'
    # # txt_dir = 'WORK_DIR//mmdetection/work_dirs/record_txt/def_detr_small_match_record.txt'
    # # txt_dir = 'WORK_DIR//mmdetection/work_dirs/record_txt/def_detr_two_refine_small_match_record.txt'
    # # parse_layer_iou(txt_dir)
    # parse_small_object_txt(txt_dir)
    parse_miss_record_txt_ = True
    parse_miss_get_record_txt_ = True
    txt_dir = 'WORK_DIR/mmdetection/work_dirs/RSVG_out_imshow/'
    if parse_miss_record_txt_:
        # txt_list = [
        #     'layer_0_grounding_dino_small_record_DOTA_iou_match_2.txt',
        #     'layer_1_grounding_dino_small_record_DOTA_iou_match_2.txt',
        #     'layer_2_grounding_dino_small_record_DOTA_iou_match_2.txt',
        #     'layer_3_grounding_dino_small_record_DOTA_iou_match_2.txt',
        #     'layer_4_grounding_dino_small_record_DOTA_iou_match_2.txt',
        #     'layer_5_grounding_dino_small_record_DOTA_iou_match_2.txt',
        #     'layer_encoder_grounding_dino_small_record_DOTA_iou_match_2.txt',
        #     ]
        txt_list = [
            'layer_5_grounding_dino_miss_record_RSVG.txt',
            'layer_5_grounding_dino_miss_IoU_match_record_RSVG.txt',
            'layer_5_grounding_dino_fusion_decouple_miss_record_RSVG.txt',
            'layer_5_grounding_dino_fusion_decouple_IoU_miss_record_RSVG.txt',
            'layer_5_grounding_dino_miss_record_RSVG_low_weight0.5.txt',
            ]
        #
        parse_miss_record_txt(txt_dir, txt_list)
    if parse_miss_get_record_txt_:
        txt_RSVG = 'layer_5_grounding_dino_miss_record_RSVG.txt'
        txt_DOTA = 'layer_5_grounding_dino_miss_record_DOTA.txt'
        parse_miss_get_record_txt(txt_dir, txt_RSVG, txt_DOTA)
