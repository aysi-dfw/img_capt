import numpy as np
import skimage.io as io
from skimage.transform import resize
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import os
import sys
import json
from mypool import MyPool
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO

# initialize COCO API for instance annotations
dataDir = '/opt/cocoapi'
dataType = 'val2014'
instances_annFile = os.path.join(dataDir, 'annotations/instances_{}.json'.format(dataType))
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
captions_annFile = os.path.join(dataDir, 'annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)

# get image ids
ids = list(coco.anns.keys())
id_list = np.random.choice(ids, size=10000)

annotations = {}


def prc_data(inp):
    idx, id_ = inp
    # load image from coco
    img_id = coco.anns[id_]['image_id']
    img = coco.loadImgs(img_id)[0]
    url = img['coco_url']

    img_file = resize(io.imread(url), (224, 224))

    # load and display captions
    annIds = coco_caps.getAnnIds(imgIds=img['id'])
    ann_dict = coco_caps.loadAnns(annIds)
    anns = [x['caption'] for x in ann_dict]

    annotations[idx] = anns
    io.imsave(os.path.join('coco_data', '{}.png'.format(str(img['id']).zfill(20))), img_as_ubyte(img_file))


pool = MyPool(20)
pool.map(prc_data, enumerate(id_list))
pool.close()
pool.join()

with open('captions.json', 'w') as f:
    json.dump(annotations, f)
