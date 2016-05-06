from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set up file names and pathes
dataDir='.'
dataType='val2014'
algName = 'fakecap'
# annFile='%s/annotations/captions_%s.json'%(dataDir,dataType)
annFile='../val_labels.json'
subtypes=['results', 'evalImgs', 'eval']
# [resFile, evalImgsFile, evalFile]= \
# ['%s/results/captions_%s_%s_%s.json'%(dataDir,dataType,algName,subtype) for subtype in subtypes]
resFile = 'val2.json'
chk = json.load(open('val.json', 'r'))
preds = chk['val_predictions'][:200]
json.dump(preds, open(resFile,'w'))


coco = COCO(annFile)
print('===== load coco done!')

# for item in coco.imgToAnns.items():
    # print(item)

# print(len(coco.imgToAnns)) ## 400 / 2z
cocoRes = coco.loadRes(resFile)

# for item in cocoRes.imgToAnns.items():
#     print(item)

print('===== load res done!')

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()

# create output dictionary
out = {}
for metric, score in cocoEval.eval.items():
    out[metric] = score
# serialize to file, to be read from Lua
json.dump(out, open('score_out.json', 'w'))