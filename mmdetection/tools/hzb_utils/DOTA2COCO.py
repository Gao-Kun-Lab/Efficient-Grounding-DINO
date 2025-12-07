import os
import cv2
import json
from PIL import Image
import sys
import codecs
import shapely.geometry as shgeo

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
wordname_Sentinel = ('Ships-moored-in-the-port', 'High-speed-ship',
               #       'Ship-on-the-sea', 'Ship-on-the-river', 'white-ship',
               # 'large-ship', 'blue-ship', 'red-ship', 'black-ship', 'small-ship',
                     )
wordname_Sentinel2 = ('The ship is moored at the dock', 'The ship is navigating',
               #       'Ship-on-the-sea', 'Ship-on-the-river', 'white-ship',
               # 'large-ship', 'blue-ship', 'red-ship', 'black-ship', 'small-ship',
                     )
wordname_Sentinel_hbb = ('ship',)
wordname_6 = ('plane', 'small-vehicle', 'large-vehicle', 'ship', 'storage-tank', 'harbor',)

wordname_9 = ('baseball-diamond', 'bridge', 'ground-track-field', 'tennis-court',
                  'basketball-court', 'soccer-ball-field', 'roundabout', 'swimming-pool', 'helicopter')

wordname_30 = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
                       'ship', 'tennis-court',
                       'basketball-court', 'storage-tank',
                       'soccer-ball-field', 'roundabout',
                       'harbor', 'swimming-pool',
                       'helicopter',
                      'plane_', 'baseball-diamond_',
                      'bridge_', 'ground-track-field_',
                      'small-vehicle_', 'large-vehicle_',
                      'ship_', 'tennis-court_',
                      'basketball-court_', 'storage-tank_',
                      'soccer-ball-field_', 'roundabout_',
                      'harbor_', 'swimming-pool_',
                      'helicopter_')
wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
                       poly[1][0], poly[1][1],
                       poly[2][0], poly[2][1],
                       poly[3][0], poly[3][1]
                       ]
    return outpoly

def parse_dota_poly2(filename):
    """
        parse the dota ground truth in the format:
        [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    objects = parse_dota_poly(filename)
    for obj in objects:
        obj['poly'] = TuplePoly2Poly(obj['poly'])
        obj['poly'] = list(map(int, obj['poly']))
    return objects

def parse_dota_poly(filename):
    """
        parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects = []
    # print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    # count = 0
    while True:
        line = f.readline()
        # count = count + 1
        # if count < 2:
        #     continue
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            #if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 9):
                continue
            if (len(splitlines) >= 9):
                    object_struct['name'] = splitlines[8]
            if (len(splitlines) == 9):
                object_struct['difficult'] = '0'
            elif (len(splitlines) >= 10):
                # if splitlines[9] == '1':
                # if (splitlines[9] == 'tr'):
                #     object_struct['difficult'] = '1'
                # else:
                object_struct['difficult'] = splitlines[9]
                # positive_map = splitlines[-1].split(',')
                # else:
                #     object_struct['difficult'] = 0
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            # poly = list(map(lambda x:np.array(x), object_struct['poly']))
            # object_struct['long-axis'] = max(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            # object_struct['short-axis'] = min(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            # if (object_struct['long-axis'] < 15):
            #     object_struct['difficult'] = '1'
            #     global small_count
            #     small_count = small_count + 1
            objects.append(object_struct)
        else:
            break
    return objects

def DOTA2COCOTrain(srcpath, destfile, cls_names, difficult='2'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'annfiles')
    print(labelparent)
    print(destfile)
    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = GetFileFromThisRootDir(labelparent)
        for i, file in enumerate(filenames):
            basename = custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = parse_dota_poly2(file)
            for obj in objects:
                if obj['difficult'] == difficult:
                    print('difficult: ', difficult)
                    continue
                if obj['name'] not in cls_names:
                    continue
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1

        json.dump(data_dict, f_out)

def Sentinel2COCOTrain(srcpath, destfile, cls_names, difficult='2'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'annfiles')
    print(labelparent)
    print(destfile)
    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = GetFileFromThisRootDir(labelparent)
        for i, file in enumerate(filenames):
            basename = custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = parse_dota_poly2(file)
            for obj in objects:
                if obj['difficult'] == difficult:
                    print('difficult: ', difficult)
                    # continue
                if obj['name'] not in wordname_Sentinel:
                    print('label {} is not in {}'.format(obj['name'], wordname_Sentinel))
                    continue

                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = wordname_Sentinel.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1

        json.dump(data_dict, f_out)

def DOTA2COCOTest(srcpath, destfile, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    data_dict = {}

    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = GetFileFromThisRootDir(imageparent)
        for i, file in enumerate(filenames):
            basename = custombasename(file)
            imagepath = os.path.join(imageparent, basename + '.png')
            img = Image.open(imagepath)
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        json.dump(data_dict, f_out)

def Sentinel2COCOTest(srcpath, destfile, txt_path, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    data_dict = {}

    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = GetFileFromThisRootDir(imageparent)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for name in lines:
                basename = name[:-5]
        # for i, file in enumerate(filenames):
        #         basename = custombasename(file)
                imagepath = os.path.join(imageparent, basename + '.png')
                img = Image.open(imagepath)
                height = img.height
                width = img.width

                single_image = {}
                single_image['file_name'] = basename + '.png'
                single_image['id'] = image_id
                single_image['width'] = width
                single_image['height'] = height
                data_dict['images'].append(single_image)

                image_id = image_id + 1
        json.dump(data_dict, f_out)

if __name__ == '__main__':

    # DOTA2COCOTrain(r'WORK_DIR/detection_data/DOTA_v1/DOTA1_1024_hbb/train/',
    #                r'WORK_DIR/detection_data/DOTA_v1/DOTA1_1024_hbb/train/DOTA_train1024.json',
    #                wordname_15)
    # DOTA2COCOTrain(r'WORK_DIR//DOTA-v1.0/dota2-split-1024/trainval1024/',
    #                r'WORK_DIR//DOTA-v1.0/dota2-split-1024/trainval1024/DOTA_trainval1024.json',
    #                wordname_15)
    # DOTA2COCOTest(r'WORK_DIR//DOTA-v1.0/dota1.5-split-1024/test1024',
    #               r'WORK_DIR//DOTA-v1.0/dota1.5-split-1024/test1024/DOTA_test1024.json',
    #               wordname_16)
    # DOTA2COCOTest(r'WORK_DIR/detection_data/DOTA_v1/DOTA1_1024_hbb/test/',
    #               r'WORK_DIR/detection_data/DOTA_v1/DOTA1_1024_hbb/test/DOTA_test1024.json',
    #               wordname_15)
    Sentinel2COCOTest(r'WORK_DIR/detection_data/datasets_Sentinel2/myself/ms_split_data_1024/testvg2/',
                      r'WORK_DIR/detection_data/datasets_Sentinel2/myself/ms_split_data_1024/testvg2/Sentinel_test1024.json',
                      r'WORK_DIR/detection_data/datasets_Sentinel2/myself/ms_split_data_1024/testvg2/text.txt',
                      wordname_Sentinel2
                      )
    # Sentinel2COCOTrain(r'WORK_DIR/detection_data/datasets_Sentinel2/myself/ms_split_data_1024/testvg2/',
    #                r'WORK_DIR/detection_data/datasets_Sentinel2/myself/ms_split_data_1024/testvg2/Sentinel_train1024.json',
    #                wordname_Sentinel2)

