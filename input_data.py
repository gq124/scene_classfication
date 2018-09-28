#!/user/bin/env python3
# -*- coding: utf-8 -*-
'''

Author:WinterFu https://github.com/Machine-Learning-For-Research/Scene_Classification_Tensorflow
'''
import os
import numpy as np
import json
import tensorflow as tf
from PIL import Image


# %%%########################################Data Path######################################

# validation_label_dir = 'E:/资料文档/研究生/研究生竞赛/AIChallenger/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
# validation_dir = 'E:/资料文档/研究生/研究生竞赛/AIChallenger/ai_challenger_scene_validation_20170908/scene_validation_images_20170908/'
# validation_save_dir = 'E:/资料文档/研究生/研究生竞赛/AIChallenger/ai_challenger_scene_validation_20170908/validation/'
#
# train_label_dir = 'E:/资料文档/研究生/研究生竞赛/AIChallenger/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'
# train_dir = 'E:/资料文档/研究生/研究生竞赛/AIChallenger/ai_challenger_scene_train_20170904/scene_train_images_20170904/'
# train_save_dir = 'E:/资料文档/研究生/研究生竞赛/AIChallenger/ai_challenger_scene_train_20170904/train/'

# %%
def get_files(label_json_dir, file_dir):  # 生成keras数据集的时候需要增加 pic_save_dir 这个参数
    """
    Args:
        label_json_dir: label.json
        file_dir:file directory
    Returns:
        list of images and labels
    """
    image_label_dict = {}
    train_image = []
    train_label = []

    # 读取label值
    with open(label_json_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 构建图片名和标签字典
    for item in data:
        image_label_dict[item['image_id']] = item['label_id']

    for file in os.listdir(file_dir):
        train_image.append(file_dir + file)
        train_label.append(image_label_dict[file])
    #######################################构建keras数据集####################################
    #    for file in os.listdir(file_dir):
    #        print(type(image_label_dict[file]))
    #        img = Image.open(file_dir + file)
    #        if int(image_label_dict[file]) in range(80):
    #            print(1)
    #            if not os.path.exists(pic_save_dir + image_label_dict[file]):
    #                os.makedirs(pic_save_dir + image_label_dict[file])
    #            img.save(pic_save_dir + image_label_dict[file]+'/' + file)
    #########################################################################################

    print("There are %d scenes" % len(train_label))

    temp = np.array([train_image, train_label])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


# %%

def get_batch(image, label, image_W, image_H, batch_size, capacipy, n_classes):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: bath size
        capacipy: the maxmum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype = tf.float32
        label_batch:  1Dtensor [batch_size], dtype = tf.int16
    '''
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make a input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    #################################################
    # data agumentation should go to here
    #################################################
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    image = tf.image.per_image_standardization(image)
    #    image_batch, label_batch = tf.train.batch([image, label],
    #                                              batch_size=batch_size,
    #                                              num_threads=64,
    #                                              capacity=capacipy)

    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=64,
                                                      capacity=capacipy,
                                                      min_after_dequeue=200)
    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.one_hot(label_batch, depth=n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int16)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])

    return image_batch, label_batch

# %% TEST

# import matplotlib.pyplot as plt
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 400
# IMG_H = 400
# label_dir = '/home/winter/TensorFlow/Scene_Classification/data/train/scene_train_annotations_20170904.json'
# train_dir = train_dir = '/home/winter/TensorFlow/Scene_Classification/data/train/scene_train_images_20170904/'
# image_list, label_list = get_files(label_dir, train_dir)
# image_batch, label_batch = get_batch(image_list,
#                                     label_list,
#                                     IMG_W, IMG_H,
#                                     BATCH_SIZE,
#                                     CAPACITY,
#                                     80)
#
# with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    try:
#        while not coord.should_stop() and i<1:
#            img, label = sess.run([image_batch, label_batch])
#
#            # just test one batch
#            for j in range(BATCH_SIZE):
#                print('label: %d' % 1)
#                print(img[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#       coord.request_stop()
#    coord.join(threads)



























