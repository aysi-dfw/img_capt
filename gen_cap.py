import os
import sys
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

annotations = {}
for file in os.listdir('coco_data'):
    if file[0] == '.':
        continue

    file = file[:-4]
    id = int(file)

    annIds = coco_caps.getAnnIds(imgIds=id)
    ann_dict = coco_caps.loadAnns(annIds)
    anns = [x['caption'] for x in ann_dict]

    annotations[id] = anns

import json
with open('captions.json', 'w') as f:
    json.dump(annotations, f)