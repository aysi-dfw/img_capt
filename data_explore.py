import numpy as np
from skimage.transform import resize
import skimage.io as io
import matplotlib.pyplot as plt
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

# get image ids
ids = list(coco.anns.keys())

print(np.shape(ids))

print(ids[0])

# pick a random image and obtain the corresponding URL
# ann_id = np.random.choice(ids, size=10)
ann_id = 1736794
img = coco.loadImgs(ann_id)[0]
url = img['coco_url']

# print URL and visualize corresponding image
print(url)
I = io.imread(url)
plt.axis('off')
plt.imshow(I)
plt.show()

io.imsave('test.png', resize(I, (224, 224)))

# load and display captions
annIds = coco_caps.getAnnIds(imgIds=img['id']);
ann_dict = coco_caps.loadAnns(annIds)
anns = [x['caption'] for x in ann_dict]
print(anns)
