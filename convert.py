# -*- coding: utf-8 -*-

import coremltools
import os

if __name__ == '__main__':
    dir = './model_alexnet/'
    coreml_model = coremltools.converters.caffe.convert((dir + 'bvlc_alexnet.caffemodel', dir + 'deploy.prototxt'),  predicted_feature_name= dir +'class_labels.txt')
    coreml_model.save('BVLCObjectClassifier.mlmodel')
    print('Model has been saved.')


