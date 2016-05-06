__author__ = 'tylin'
__version__ = '1.0.1'
# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  segToMask  - Convert polygon segmentation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load result file and create result api object.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>segToMask, COCO>showAnns

# Microsoft COCO Toolbox.      Version 1.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from skimage.draw import polygon
import copy

class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = {}
        self.imgToAnns = {}

        if not annotation_file == None:
            print 'loading annotations into memory...'
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            print datetime.datetime.utcnow() - time_t
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print 'creating index...'
        imgToAnns = {ann['image_id']: [] for ann in self.dataset}
        for ann in self.dataset:
            imgToAnns[ann['image_id']] += [ann]
        print 'index created!'
        self.imgToAnns = imgToAnns
        print(type(self.imgToAnns))
        # for id, ann in enumerate(self.imgToAnns.items()): ## change keys to index numbers
        #     self.imgToAnns[id] = self.imgToAnns.pop(ann[0])

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''

        ids = self.imgToAnns.keys()
        return list(ids)


    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset = [img for img in self.dataset] ## copy self.dataset to res.dataset

        print 'Loading and preparing results...     '
        time_t = datetime.datetime.utcnow()
        anns    = json.load(open(resFile))
        # anns = anns['val_predictions']
        # anns = anns[:200]
        # print(anns)
        # print(len(anns))
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]

        imgIds = set([img['image_id'] for img in res.dataset]) & set([ann['image_id'] for ann in anns])
        print('---------- gnd label length = ')
        print(len(set([img['image_id'] for img in res.dataset]))) ## gnd labels
        print('---------- output data length = ')
        print(len(set([ann['image_id'] for ann in anns]))) ## output labels
        print('---------- overlapping data length = ')
        print(len(imgIds))
        res.dataset = [img for img in res.dataset if img['image_id'] in imgIds]
        # print(len(res.dataset))
        # for id, ann in enumerate(anns):
        #     ann['id'] = id
        print 'DONE (t=%0.2fs)'%((datetime.datetime.utcnow() - time_t).total_seconds())

        res.dataset = anns
        res.createIndex()
        return res