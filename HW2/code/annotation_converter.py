import os
import glob
import h5py
import numpy as np
import cv2

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    return attrs

if __name__ == '__main__':
    path = os.getcwd()
    mat = 'dataset/train/digitStruct.mat'
    mat_path = os.path.join(path, mat)
    annotation = h5py.File(mat_path, 'r') 

    # Training data
    train_size = annotation['digitStruct']['name'].shape[0]
    traintxt_path = os.path.join(path, 'darknet-master/data/train_obj.txt')
    train_f = open(traintxt_path, 'w')

    for idx in range(train_size):
        name = get_name(idx, annotation)
        bbox = get_bbox(idx, annotation)

        if idx % 500 == 0:
            print(name)

        train_f.write('data/obj/train/' + name + '\n')

        imgtxt_path = os.path.join(path, 'darknet-master/data/obj/train/{0}.txt'.format(name.split('.', 1)[0]))
        img_f = open(imgtxt_path, 'w')

        img_path = os.path.join(path, 'darknet-master/data/obj/train', name)
        image = cv2.imread(img_path)
        h, w, _ = image.shape

        for i in range(len(bbox['label'])):
            object_class = str(int(bbox['label'][i]))
            x_center = str((bbox['left'][i] + (bbox['width'][i] / 2)) / w)
            y_center = str((bbox['top'][i] + (bbox['height'][i] / 2)) / h)
            width = str(bbox['width'][i] / w)
            height = str(bbox['height'][i] / h)
            img_f.write(object_class + ' ' + x_center + ' ' + y_center + ' ' + width + ' ' + height + '\n')
        img_f.close()

    train_f.close()

    # Test data
    testtxt_path = os.path.join(path, 'darknet-master/data/test_obj.txt')
    test_f = open(testtxt_path, 'w')
    test_path = os.path.join(path, 'darknet-master/data/obj/test/*.png')
    test_img = glob.glob(test_path)
    test_size = len(test_img)
    print(test_size)

    for test_name in test_img:
        test_name = test_name.rsplit('/', 1)[-1]
        test_f.write('data/obj/test/' + test_name + '\n')

    test_f.close()