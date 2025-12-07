import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

def cam_pred_bg_analyze(data_dir, txt_list):
    delta_all = [[[], [], [], []] for _ in txt_list]

    # read txt and get data
    for i, txt_name in enumerate(txt_list):
        with open(osp.join(data_dir, txt_name), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                line = [abs(float(_)) for _ in line]
                delta_all[i][0].append(line[0])
                delta_all[i][1].append(line[1])
                delta_all[i][2].append(line[2])
                delta_all[i][3].append(line[3])
            f.close()

    # draw figure
    x = [_ for _ in range(len(delta_all[0][0]))]
    # for i in range(4):
    #     plt.figure()
    #     colors = ['red', 'green', 'blue', 'purple', 'orange']
    #
    #     for j in range(len(txt_list)):
    #         plt.scatter(x, sorted(delta_all[j][i]), c=colors[j], label=txt_list[j].split('.')[0])
    #     plt.title('layer {}'.format(i))
    #     plt.xlabel('Index')
    #     plt.ylabel('Delta')
    #
    #     plt.legend()
    #
    #     # 显示图表
    #     plt.show()
    #     plt.close()

    for i in range(len(txt_list)):
        plt.figure()
        colors = ['red', 'green', 'blue', 'purple', 'orange']

        for j in range(4):
            plt.scatter(x, sorted(delta_all[i][j]), c=colors[j], label='layer {}'.format(j))
        plt.title(txt_list[i].split('.')[0])
        plt.xlabel('Index')
        plt.ylabel('Delta')

        plt.legend()

        # 显示图表
        plt.show()
        plt.close()
    # numpy_data = np.stack([layer1, layer2, layer3, layer4]).transpose(1,0)
    # labels = ['layer1', 'layer2', 'layer3', 'layer4']
    # fig, ax = plt.subplots()
    # ax.boxplot(numpy_data, labels=labels)
    # plt.legend()
    # plt.show()
    # x = [_ for _ in range(len(layer1))]
    # plt.figure()
    # plt.scatter(x, layer1)
    # plt.show()
    # plt.close()
    # plt.scatter(x, layer2)
    # plt.show()
    # plt.close()
    #
    # plt.scatter(x, layer3)
    # plt.show()
    # plt.close()
    #
    # plt.scatter(x, layer4)
    # plt.show()



if __name__=='__main__':
    data_dir = 'WORK_DIR/mmdetection/work_dirs/RSVG_out_imshow/'
    txt_list = [
                "gt_bg_trainval_delta_median_grounding_dino.txt",
                "gt_bg_trainval_delta_median_fusion_decouple.txt"
                # "gt_bg_delta_median_grounding_dino.txt",
                # "gt_bg_delta_median_fusion_decouple.txt"
                # 'gt_bg_delta_grounding_dino_back.txt',
                # 'gt_bg_delta_fusion_decouple_back.txt',
                # 'gt_bg_delta_LQVG_back.txt',
                # 'gt_bg_delta_decouple_ms_back.txt',
                # 'gt_bg_delta_decouple_text_single_back.txt'
                ]
    # txt_path = 'WORK_DIR/mmdetection/work_dirs/RSVG_out_imshow/gt_bg_delta_grounding_dino_back.txt'
    # txt_path = 'WORK_DIR/mmdetection/work_dirs/RSVG_out_imshow/gt_bg_delta_fusion_decouple_back.txt'
    # txt_path = 'WORK_DIR/mmdetection/work_dirs/RSVG_out_imshow/gt_bg_delta_LQVG_back.txt'
    # txt_path = 'WORK_DIR/mmdetection/work_dirs/RSVG_out_imshow/gt_bg_delta_decouple_ms_back.txt'
    # txt_path = 'WORK_DIR/mmdetection/work_dirs/RSVG_out_imshow/gt_bg_delta_decouple_text_single_back.txt'
    cam_pred_bg_analyze(data_dir, txt_list)

