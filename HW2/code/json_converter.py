import os
import json
import cv2

def extract_img(json):
    try:
        # Also convert to int since update_time will be string.  When comparing
        # strings, "10" is smaller than "2".
        return int(json['filename'].rsplit('/', 1)[-1].split('.', 1)[0])
    except KeyError:
        return 0



if __name__ == '__main__':
    with open('result.json' , 'r') as reader:
        jf = json.loads(reader.read())

    jf.sort(key=extract_img, reverse=False)

    sub = []
    for idx in jf:
        dict_ = {'bbox': [], 'score': [], 'label': []}

        path = os.getcwd()
        img_path = os.path.join(path, 'darknet-master', idx['filename'])
        image = cv2.imread(img_path)
        h, w, _ = image.shape

        for o in idx['objects']:
            center_x = o['relative_coordinates']['center_x'] * w
            center_y = o['relative_coordinates']['center_y'] * h
            width = o['relative_coordinates']['width'] * w
            height = o['relative_coordinates']['height'] * h

            y1 = center_y - (height / 2)
            x1 = center_x - (width / 2)
            y2 = center_y + (height / 2)
            x2 = center_x + (width / 2)

            dict_['bbox'].append((y1, x1, y2, x2))
            dict_['score'].append(o['confidence'])
            if o['class_id'] == 0:
                dict_['label'].append(10)
            else:
                dict_['label'].append(o['class_id'])

        sub.append(dict_)

    print(len(sub))
    ret = json.dumps(sub)
    with open('0856087.json', 'w') as fp:
        fp.write(ret)