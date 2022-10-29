import numpy as np

datasets=[ 'train',  'val',  'test']


for image_set in datasets:
    image_ids = open('data/datasets/%s.txt' % image_set).read().strip().split()
    list_file = open('%s.txt' % image_set, 'w')
    for image_id in image_ids:
        list_file.write('data/image/%s.tif ' % image_id)
        list_file.write('data/label/%s.tif' % image_id)
        list_file.write('\n')

    list_file.close()





