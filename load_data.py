import numpy as np

import os
import urllib
import gzip
from scipy import misc


def data_generator(batch_size, data_dir):
    images = []
    for filename in os.listdir(data_dir):
        img = misc.imread(filename)
        images.append(img)

    #images = np.concatenate(all_data, axis=0)

    def get_epoch():
        np.random.shuffle(images)

        for i in xrange(len(images) / batch_size):
            yield np.copy(images[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        data_generator(batch_size, data_dir), 
        data_generator(batch_size, data_dir)
    )