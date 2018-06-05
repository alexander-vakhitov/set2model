import caffe, numpy as np, yaml
import loader
import os
import time

class LoadImagesLayer(caffe.Layer):


    def check_params(self, params):
        #assert 'split' in params.keys(), 'Params must include split (train, val, or test).'
        required = ['db_dir', 'im_shape', 'max_query_size',
                    'pos_test_num', 'neg_test_num', 'do_load_negs', 'q_path', 't_path', 'group_cnt']
        for r in required:
            assert r in params.keys(), 'Params must include {}'.format(r)



    def setup(self, bottom, top):

        self.top_names = ['data', 'label', 'query_size']
        # params is a python dictionary with layer parameters.
        params = yaml.load(self.param_str)
        # Check the paramameters for validity.
        self.check_params(params)
        db_dir = params['db_dir']
        lmdb_db_path_q, lmdb_db_path_t, meta_db_path_q, meta_db_path_t = self.define_db_names(db_dir)

        self.max_query_size = params['max_query_size']
        self.im_shape = params['im_shape']
        self.pos_test_num = params['pos_test_num']
        self.neg_test_num = params['neg_test_num']
        caffe_root = params['caffe_root']
        self.group_cnt = params['group_cnt']

        self.filterQFile = ''
        if (os.path.exists(db_dir+'/dup')):
            self.filterQFile = db_dir+'/dup'

        self.q_loader = loader.Loader(lmdb_db_path_q, meta_db_path_q, caffe_root, self.im_shape[1], filterQFile=self.filterQFile)
        self.q_loader.set_group_cnt(self.group_cnt)

        if params['do_load_negs']:
            self.t_loader = loader.Loader(lmdb_db_path_t, meta_db_path_t, caffe_root, self.im_shape[1], do_load_stats=True)
            self.t_loader.load_stats(params['q_path'], params['t_path'], params['group_cnt'])
        else:
            self.t_loader = loader.Loader(lmdb_db_path_t, meta_db_path_t, caffe_root, self.im_shape[1])

        self.t_loader.set_group_cnt(self.group_cnt)

        self.batch_size = self.pos_test_num  + self.neg_test_num + self.max_query_size

        #data
        top[0].reshape(self.batch_size, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        #label
        top[1].reshape(self.batch_size)
        #query size
        top[2].reshape(1)
        #current group id
        top[3].reshape(1)

    def define_db_names(self, db_dir):
        lmdb_db_path_q = db_dir + '/' + 'testq'
        meta_db_path_q = db_dir + '/' + 'meta_test_q'
        lmdb_db_path_t = db_dir + '/' + 'testt'
        meta_db_path_t = db_dir + '/' + 'meta_test_t'
        return lmdb_db_path_q, lmdb_db_path_t, meta_db_path_q, meta_db_path_t

    def forward(self, bottom, top):
        s1 = time.time()
        s1all = time.time()

        rand_group_id = -1
        t_coll_pos = []
        t_coll_neg = []
        q_coll = []
        pos_ids = []
        neg_ids = []
        q_ids = []

        while (len(t_coll_pos) < self.pos_test_num or len(q_coll) == 0 or len(t_coll_neg)<self.neg_test_num):
            rand_group_id = np.random.randint(0, self.group_cnt)
            t_coll_pos, pos_ids = self.t_loader.get_image_collection(rand_group_id, N=self.pos_test_num)
            t_coll_neg, neg_ids = self.t_loader.get_image_collection(rand_group_id, does_belong=False, N=self.neg_test_num)
            q_coll, q_ids = self.q_loader.get_image_collection(rand_group_id, N=self.max_query_size)

        for i in range(0, len(t_coll_pos)):
            top[0].data[i, :, :, :] = t_coll_pos[i]
            top[1].data[i] = 1
        for i in range(len(t_coll_pos), self.pos_test_num):
            top[0].data[i, :, :, :] = np.zeros((self.im_shape[0], self.im_shape[1], self.im_shape[2]))
            top[1].data[i] = -1


        for i in range(0, len(t_coll_neg)):
            top[0].data[i+self.pos_test_num, :, :, :] = t_coll_neg[i]
            top[1].data[i+self.pos_test_num] = 0

        for i in range(len(t_coll_neg), self.neg_test_num):
            top[0].data[i+self.pos_test_num, :, :, :] = np.zeros((self.im_shape[0], self.im_shape[1], self.im_shape[2]))
            top[1].data[i+self.pos_test_num] = -1



        test_shift = self.pos_test_num + self.neg_test_num



        for i in range(0, len(q_coll)):
            top[0].data[i+test_shift, ...] = q_coll[i]
            top[1].data[i+test_shift, ...] = 2
        for i in range(len(q_coll), self.max_query_size):
            top[0].data[i+test_shift, ...] = np.zeros((self.im_shape[0], self.im_shape[1], self.im_shape[2]))
            top[1].data[i+test_shift, ...] = -1

        top[3].data[0] = rand_group_id


    def reshape(self, bottom, top):
        # data
        top[0].reshape(self.batch_size, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        # label
        top[1].reshape(self.batch_size)
        # current group id
        top[2].reshape(1)



    def backward(self, top, propagate_down, bottom):
        pass



class LoadImagesOmniglotLayer(caffe.Layer):

    def check_params(self, params):
        #assert 'split' in params.keys(), 'Params must include split (train, val, or test).'
        required = ['db_dir', 'batch_size', 'caffe_root', 'batch_size', 'group_cnt', 'im_shape', 'set_triplet_size']
        for r in required:
            assert r in params.keys(), 'Params must include {}'.format(r)


    def reshape(self, bottom, top):
        # data
        top[0].reshape(self.batch_size, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        #rotation_label
        top[1].reshape(self.batch_size)
        #current group id
        n_groups = self.batch_size / self.set_triplet_size
        top[2].reshape(n_groups)


    def setup(self, bottom, top):
        params = yaml.load(self.param_str)
        self.check_params(params)
        caffe_root = params['caffe_root']
        db_dir = params['db_dir']
        self.group_cnt = params['group_cnt']
        self.batch_size = params['batch_size']
        self.set_triplet_size = params['set_triplet_size']
        self.im_shape = params['im_shape']

        lmdb_db_path = db_dir + '/' + 'test_all'
        meta_db_path = db_dir + '/' + 'meta_test_all'
        self.loader = loader.Loader(lmdb_db_path, meta_db_path, caffe_root, self.im_shape[1])
        self.group_id = 0



    def forward(self, bottom, top):
        # while (self.group_id < 309 or self.load_from_q_base):

        n_groups = self.batch_size / self.set_triplet_size
        for gi in range(0, n_groups):
            self.group_id = np.random.randint(0, self.group_cnt)
            all_imgs_lst, ids = self.loader.get_image_collection(self.group_id, do_transform=False)
            all_imgs_lst = np.random.permutation(all_imgs_lst)
            all_imgs_mat = np.asarray(all_imgs_lst)
            img_num = len(all_imgs_lst)
            set_size = self.set_triplet_size/3 #img_num / 2
            neg_imgs_lst, ids = self.loader.get_image_collection(self.group_id, do_transform=False, does_belong=False, N = set_size)
            neg_imgs_mat = np.asarray(neg_imgs_lst)
            rand_rot_ind_train = np.random.randint(0, 4)
            rand_rot_ind_pos = rand_rot_ind_train
            rand_rot_inds_neg = np.random.randint(0, 4, set_size)
            rand_rot_inds = np.concatenate([rand_rot_ind_pos*np.ones(2*set_size), rand_rot_inds_neg])
            top[0].data[self.set_triplet_size * gi: self.set_triplet_size*gi+set_size, :] = all_imgs_mat[0:set_size, :]
            top[0].data[self.set_triplet_size * gi + set_size: self.set_triplet_size * gi + 2*set_size, :] = all_imgs_mat[set_size:2*set_size, :]
            top[0].data[self.set_triplet_size * gi + 2*set_size: self.set_triplet_size * gi + 3 * set_size, :] = neg_imgs_mat
            top[1].data[self.set_triplet_size*gi: self.set_triplet_size*(gi+1)] = rand_rot_inds
            top[2].data[gi] = self.group_id


class LoadImagesMLOmniglotLayer(caffe.Layer):

    def check_params(self, params):
        #assert 'split' in params.keys(), 'Params must include split (train, val, or test).'
        required = ['db_dir', 'batch_size', 'caffe_root', 'batch_size', 'group_cnt', 'im_shape', 'set_triplet_size']
        for r in required:
            assert r in params.keys(), 'Params must include {}'.format(r)


    def reshape(self, bottom, top):
        # data
        top[0].reshape(self.batch_size, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        #rotation_label
        top[1].reshape(self.batch_size)
        #current group id
        n_groups = self.batch_size / self.set_triplet_size
        top[2].reshape(n_groups)
        # print 'reshape done '


    def setup(self, bottom, top):
        params = yaml.load(self.param_str)
        self.check_params(params)
        caffe_root = params['caffe_root']
        db_dir = params['db_dir']
        self.group_cnt = params['group_cnt']
        self.batch_size = params['batch_size']
        self.set_triplet_size = params['set_triplet_size']
        self.im_shape = params['im_shape']
        for key in params:
            print '|'+key+'|'
        if ('train_set_size' in params):
            self.train_set_size = params['train_set_size']
            self.test_set_size = params['test_set_size']
            self.test_pos_size = params['test_pos_size']
            self.test_neg_size = self.test_set_size - self.test_pos_size
        else:
            self.train_set_size = -1
            self.test_set_size = -1
            self.test_pos_size = -1
            self.test_neg_size = -1

        if ('meta_mode' in params):
            self.meta_mode = True
            print 'META MODE ENABLED'
        else:
            self.meta_mode = False
            print 'META MODE DISABLED'

        lmdb_db_path = db_dir + '/' + 'test_all'
        meta_db_path = db_dir + '/' + 'meta_test_all'
        self.loader = loader.Loader(lmdb_db_path, meta_db_path, caffe_root, self.im_shape[1])
        self.update_group_ids()


        self.group_id = 0

    def update_group_ids(self):
        n_groups = self.batch_size / self.set_triplet_size
        self.group_ids = []
        self.class_rots = []
        for gi in range(0, n_groups):
            self.group_ids.append(np.random.randint(0, self.group_cnt))
            self.class_rots.append(np.random.randint(0, 4))

    def form_not_in_lst(self, set_size, class_id, rot_id):
        img_lst = []
        rot_lst = []
        while (len(img_lst) < set_size):
            rand_gi = np.random.randint(0, self.group_cnt)
            rand_roti = np.random.randint(0, 4)
            if (rand_gi != class_id or rot_id != rand_roti):
                neg_imgs_lst, ids = self.loader.get_image_collection(rand_gi, do_transform=False, does_belong=True,
                                                                     N=1)
                rot_lst.append(rand_roti)
                img_lst += neg_imgs_lst


        return img_lst, rot_lst




    def forward(self, bottom, top):
        if (not self.meta_mode):
            self.update_group_ids()
        # while (self.group_id < 309 or self.load_from_q_base):

        n_groups = self.batch_size / self.set_triplet_size
        for gi in range(0, n_groups):
            self.group_id = self.group_ids[gi]
            all_imgs_lst, ids = self.loader.get_image_collection(self.group_id, do_transform=False)
            all_imgs_lst = np.random.permutation(all_imgs_lst)
            all_imgs_mat = np.asarray(all_imgs_lst)
            img_num = len(all_imgs_lst)
            if (self.train_set_size < 0):

                set_size = self.set_triplet_size/3 #img_num / 2
                # neg_imgs_lst, ids = self.loader.get_image_collection(self.group_id, do_transform=False, does_belong=False, N = set_size)
                neg_imgs_lst, neg_rot_lst = self.form_not_in_lst(set_size, self.group_id, self.class_rots[gi])
                neg_imgs_mat = np.asarray(neg_imgs_lst)
                # top[0].data[self.set_triplet_size * gi: self.set_triplet_size*gi+set_size, :] = all_imgs_mat[0:set_size, :]
                # top[0].data[self.set_triplet_size * gi + set_size: self.set_triplet_size * gi + 2*set_size, :] = all_imgs_mat[set_size:2*set_size, :]
                top[0].data[self.set_triplet_size * gi: self.set_triplet_size * gi + 2*set_size, :] = all_imgs_mat[
                                                                                                    0:2*set_size, :]
                top[0].data[self.set_triplet_size * gi + 2*set_size: self.set_triplet_size * gi + 3 * set_size, :] = neg_imgs_mat

                top[1].data[self.set_triplet_size * gi: self.set_triplet_size*gi+2*set_size] = self.class_rots[gi]
                top[1].data[self.set_triplet_size * gi+2*set_size: self.set_triplet_size * (gi + 1)] = np.asarray(neg_rot_lst)

            else:
                # neg_imgs_lst, ids = self.loader.get_image_collection(self.group_id, do_transform=False, does_belong=False, N = self.test_neg_size)
                neg_imgs_lst, neg_rot_lst = self.form_not_in_lst(self.test_neg_size, self.group_id, self.class_rots[gi])
                neg_imgs_mat = np.asarray(neg_imgs_lst)
                if (neg_imgs_mat.shape[0] == 0):
                    neg_imgs_mat = neg_imgs_mat.reshape(0, self.im_shape[0], self.im_shape[1], self.im_shape[2])
                cp = self.set_triplet_size * gi
                top[0].data[cp: cp + self.train_set_size, :] = all_imgs_mat[0:self.train_set_size, :]
                top[0].data[cp + self.train_set_size: cp + self.train_set_size + self.test_pos_size,
                    :] = all_imgs_mat[self.train_set_size: self.train_set_size + self.test_pos_size, :]
                top[0].data[cp + self.train_set_size + self.test_pos_size: cp + self.train_set_size + self.test_set_size,
                    :] = neg_imgs_mat

                top[1].data[cp:cp+self.train_set_size + self.test_pos_size] = self.class_rots[gi]
                top[1].data[cp+self.train_set_size + self.test_pos_size : self.set_triplet_size * (gi+1)] = np.asarray(neg_rot_lst)




            # rand_rot_inds = np.random.randint(0, 4, self.set_triplet_size)
            # top[1].data[self.set_triplet_size * gi: self.set_triplet_size * (gi + 1)] = rand_rot_inds
            top[2].data[gi] = self.group_id




class LoadImagesOmniglotDiscrimLayer(caffe.Layer):

    def check_params(self, params):
        #assert 'split' in params.keys(), 'Params must include split (train, val, or test).'
        required = ['db_dir', 'caffe_root', 'im_shape', 'group_cnt', 'class_num', 'train_per_class', 'test_per_class', 'batch_len']
        for r in required:
            assert r in params.keys(), 'Params must include {}'.format(r)


    def reshape(self, bottom, top):
        # data
        batch_size = self.batch_len*(self.group_size)
        top[0].reshape(batch_size, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        #rotation_label
        top[1].reshape(batch_size)



    def setup(self, bottom, top):
        params = yaml.load(self.param_str)
        self.check_params(params)
        caffe_root = params['caffe_root']
        db_dir = params['db_dir']
        self.group_cnt = params['group_cnt']
        self.class_num = params['class_num']
        self.train_per_class = params['train_per_class']
        self.test_per_class = params['test_per_class']
        self.batch_len = params['batch_len']
        self.im_shape = params['im_shape']

        lmdb_db_path = db_dir + '/' + 'test_all'
        meta_db_path = db_dir + '/' + 'meta_test_all'
        self.loader = loader.Loader(lmdb_db_path, meta_db_path, caffe_root, self.im_shape[1])
        self.group_size = self.class_num * self.train_per_class + self.class_num * self.test_per_class

#train set, then test set
    def forward(self, bottom, top):

        for gi in range(0, self.batch_len):
            class_ids = np.random.permutation(self.group_cnt)[0:self.class_num]
            for ci in range(0, self.class_num):
                all_imgs_lst, ids = self.loader.get_image_collection(class_ids[ci],
                                                                     do_transform=False,
                                                                     N = self.test_per_class + self.train_per_class,
                                                                     w_rep = True)
                all_imgs_mat = np.asarray(all_imgs_lst)
                top[0].data[self.group_size*gi + ci*self.train_per_class :
                    self.group_size*gi + (ci+1)*self.train_per_class, :] = all_imgs_mat[0:self.train_per_class, :]
                grshift = self.group_size*gi + self.class_num*self.train_per_class
                top[0].data[grshift + ci * self.test_per_class:
                    grshift + (ci + 1)*self.test_per_class, :] = all_imgs_mat[self.train_per_class:, :]

        rand_rot_inds = np.random.randint(0, 4, self.batch_len * self.group_size)
        top[1].data[:] = rand_rot_inds



class MetaLoaderOmniglotLayer(caffe.Layer):

    def check_params(self, params):
        #assert 'split' in params.keys(), 'Params must include split (train, val, or test).'
        required = ['db_dir', 'caffe_root', 'im_shape', 'group_cnt', 'class_num', 'train_per_class', 'test_per_class', 'batch_len']
        for r in required:
            assert r in params.keys(), 'Params must include {}'.format(r)


    def reshape(self, bottom, top):
        # data
        batch_size = self.batch_len*(self.group_size)
        top[0].reshape(batch_size, self.im_shape[0], self.im_shape[1], self.im_shape[2])
        #rotation_label
        top[1].reshape(batch_size)
        # print 'reshape done '


    def setup(self, bottom, top):
        params = yaml.load(self.param_str)
        self.check_params(params)
        caffe_root = params['caffe_root']
        db_dir = params['db_dir']
        self.group_cnt = params['group_cnt']
        self.class_num = params['class_num']
        self.train_per_class = params['train_per_class']
        self.test_per_class = params['test_per_class']
        self.batch_len = params['batch_len']
        self.im_shape = params['im_shape']

        lmdb_db_path = db_dir + '/' + 'test_all'
        meta_db_path = db_dir + '/' + 'meta_test_all'
        self.loader = loader.Loader(lmdb_db_path, meta_db_path, caffe_root, self.im_shape[1])
        self.meta_loader = loader.MetaLoader(self.loader, self.group_cnt, self.class_num,
                                             self.test_per_class, self.train_per_class)
        self.group_size = self.class_num * self.train_per_class + self.class_num * self.test_per_class

        self.reshape(bottom, top)

    def get_meta_loader(self):
        return self.meta_loader

#train set, then test set
    def forward(self, bottom, top):
        train_batch, train_rots = self.meta_loader.get_train_batch()
        test_batch, test_rots = self.meta_loader.get_test_batch()
        for ci in range(0, len(train_batch)):
            class_imgs = train_batch[ci]
            train_imgs_mat = np.asarray(class_imgs)
            train_imgs_mat = train_imgs_mat.reshape(-1, self.im_shape[0], self.im_shape[1], self.im_shape[2])
            top[0].data[ci*self.train_per_class :
                (ci+1)*self.train_per_class, :] = train_imgs_mat

            test_imgs_mat = np.asarray(test_batch[ci])
            test_imgs_mat = test_imgs_mat.reshape(-1, self.im_shape[0], self.im_shape[1], self.im_shape[2])
            grshift = self.class_num*self.train_per_class
            top[0].data[grshift + ci * self.test_per_class:
                grshift + (ci + 1)*self.test_per_class, :] = test_imgs_mat

            top[1].data[ci*self.train_per_class :
                (ci+1)*self.train_per_class] = train_rots[ci*self.train_per_class :
                (ci+1)*self.train_per_class]
            top[1].data[grshift + ci * self.test_per_class:
            grshift + (ci + 1) * self.test_per_class] = test_rots[ci * self.test_per_class:(ci + 1)*self.test_per_class]

        # rand_rot_inds = np.random.randint(0, 4, self.group_size)
