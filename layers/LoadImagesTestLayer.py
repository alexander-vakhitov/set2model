import caffe, lmdb, numpy as np, yaml
import Debugger as dbg
import loader
import os
import cv2

import RandomRotationLayer as rrl

class LoadImagesTestLayer(caffe.Layer):

    def check_params(self, params):
        #assert 'split' in params.keys(), 'Params must include split (train, val, or test).'
        required = ['db_dir', 'batch_size', 'caffe_root', 'batch_size', 'group_cnt', 'im_shape']
        for r in required:
            assert r in params.keys(), 'Params must include {}'.format(r)


    def reshape(self, bottom, top):
        # data
        top[0].reshape(self.batch_size, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        #label
        top[1].reshape(self.batch_size)
        #data type
        top[2].reshape(1)
        #current group id
        top[3].reshape(1)
        print 'reshape done '




    def setup(self, bottom, top):

        params = yaml.load(self.param_str)
        self.check_params(params)

        caffe_root = params['caffe_root']
        db_dir = params['db_dir']
        self.group_cnt = params['group_cnt']

        self.batch_size = params['batch_size']
        lmdb_db_path_q, lmdb_db_path_t, meta_db_path_q, meta_db_path_t = self.define_db_paths(db_dir)

        self.im_shape = params['im_shape']

        self.q_loader = loader.Loader(lmdb_db_path_q, meta_db_path_q, caffe_root, self.im_shape[1])
        self.t_loader = loader.Loader(lmdb_db_path_t, meta_db_path_t, caffe_root, self.im_shape[1])



        self.load_from_q_base = True
        self.is_finished = False
        self.group_id = 0
        self.item_id = 0
        #data
        top[0].reshape(self.batch_size, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        #labels
        top[1].reshape(self.batch_size)
        #group
        top[2].reshape(1)
        #database (query or test)
        top[3].reshape(1)

    def define_db_paths(self, db_dir):
        lmdb_db_path_q = db_dir + '/' + 'testq'
        meta_db_path_q = db_dir + '/' + 'meta_test_q'
        lmdb_db_path_t = db_dir + '/' + 'testt'
        meta_db_path_t = db_dir + '/' + 'meta_test_t'
        return lmdb_db_path_q, lmdb_db_path_t, meta_db_path_q, meta_db_path_t

    def forward(self, bottom, top):

        # while (self.group_id < 309 or self.load_from_q_base):

        if (self.is_finished):
            top[0].data[...] = np.zeros_like(top[0].data)
            top[1].data[...] = -1*np.ones_like(top[1].data)
            top[2].data[...] = np.zeros_like(top[2].data)
            top[3].data[...] = np.zeros_like(top[3].data)
            return


        imgs = []
        lbls = []
        do_switch_db = False

        group_seld = self.group_id
        if (self.load_from_q_base):
            all_imgs, ids = self.q_loader.get_image_collection(self.group_id)
            imgs = all_imgs[self.item_id:self.item_id + self.batch_size]
            lbls = np.arange(self.item_id, self.item_id + len(imgs))
            do_switch_db = self.update_counters(all_imgs)
        else:
            all_imgs, ids = self.t_loader.get_image_collection(self.group_id)
            imgs = all_imgs[self.item_id:self.item_id + self.batch_size]
            lbls = np.arange(self.item_id, self.item_id + len(imgs))
            do_switch_db = self.update_counters(all_imgs)

        for i in range(0, len(imgs)):
            top[0].data[i, ...] = imgs[i]
            top[1].data[i] = lbls[i]

        for i in range(len(imgs), self.batch_size):
            top[0].data[i, ...] = np.zeros((top[0].data.shape[1], top[0].data.shape[2], top[0].data.shape[3]))
            top[1].data[i] = -1

        top[2].data[0] = group_seld
        top[3].data[0] = 0
        if (self.load_from_q_base):
            top[3].data[0] = 1

        if (do_switch_db):
            self.group_id = 0
            self.item_id = 0
            if (self.load_from_q_base):
                self.load_from_q_base = False
            else:
                self.is_finished = True

        # f_out = open('/home/avakhitov/testloadlog.txt', 'a')
        # stage = 'q'
        # if (not self.load_from_q_base):
        #     stage = 't'
        # f_out.write('cur ' + str(self.group_id) + ' '+stage + ' \n')
        # f_out.close()
        #
        # f_out = open('/home/avakhitov/testloadlog.txt', 'a')
        # f_out.write('written '+str(self.group_id)+'\n')
        # f_out.close()


    def update_counters(self, all_imgs):
        do_switch_db = False
        if (self.item_id + self.batch_size > len(all_imgs)):
            self.group_id += 1
            self.item_id = 0
            if (self.group_id == self.group_cnt):
                do_switch_db = True
        else:
            self.item_id += self.batch_size
        return do_switch_db

    def backward(self, top, propagate_down, bottom):
        pass


class LoadImagesTestOmniglotLayer(caffe.Layer):

    def check_params(self, params):
        #assert 'split' in params.keys(), 'Params must include split (train, val, or test).'
        required = ['db_dir', 'batch_size', 'caffe_root', 'batch_size', 'group_cnt', 'im_shape']
        for r in required:
            assert r in params.keys(), 'Params must include {}'.format(r)


    def reshape(self, bottom, top):
        # data
        top[0].reshape(self.batch_size, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        #rotation_label
        top[1].reshape(self.batch_size)
        #current group id
        top[2].reshape(1)
        #'query' data (for descriptorsaver)
        top[3].reshape(1)
        print 'reshape done '


    def setup(self, bottom, top):
        params = yaml.load(self.param_str)
        self.check_params(params)
        caffe_root = params['caffe_root']
        db_dir = params['db_dir']
        self.group_cnt = params['group_cnt']
        self.batch_size = params['batch_size']
        self.im_shape = params['im_shape']

        lmdb_db_path = db_dir + '/' + 'test_all'
        meta_db_path = db_dir + '/' + 'meta_test_all'
        self.loader = loader.Loader(lmdb_db_path, meta_db_path, caffe_root, self.im_shape[1])

        self.group_id = 0
        self.is_finished = False

    def forward(self, bottom, top):
        # while (self.group_id < 309 or self.load_from_q_base):

        if (self.is_finished):
            top[0].data[...] = np.zeros_like(top[0].data)
            top[1].data[...] = -1 * np.ones_like(top[1].data)
            top[2].data[...] = np.zeros_like(top[2].data)
            top[3].data[:] = np.ones_like(top[3].data)
            return

        all_imgs_lst, ids = self.loader.get_image_collection(self.group_id, do_transform=False)
        all_imgs_mat = np.asarray(all_imgs_lst)
        img_num = len(all_imgs_lst)
        top[1].data[:] = -1* np.ones_like(top[1].data)
        for rot_ind in range(0, 4):
            top[0].data[rot_ind*img_num : (rot_ind+1)*img_num, :] = all_imgs_mat
            top[1].data[rot_ind*img_num : (rot_ind+1)*img_num] = rot_ind*np.ones(img_num)

        top[2].data[:] = self.group_id
        top[3].data[:] = 1

        self.group_id += 1
        if (self.group_id == self.group_cnt):
            self.is_finished = True

