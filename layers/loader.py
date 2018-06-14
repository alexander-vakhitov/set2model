import caffe, lmdb, numpy as np, time

class MetaLoader:

    def __init__(self, loader, class_num, class_in_batch_num, test_samples_per_class_num, train_samples_per_class_num):
        self.loader = loader
        self.class_num = class_num
        self.class_in_batch_num = class_in_batch_num
        self.test_samples_per_class = test_samples_per_class_num
        self.train_samples_per_class = train_samples_per_class_num
        self.sample_batches()

    def sample_batches(self):
        self.class_ids = np.random.permutation(self.class_num)[0:self.class_in_batch_num]
        self.class_rots = np.random.randint(0, 4, self.class_num)
        self.train_batch = self.prepare_batch(self.class_ids, self.class_rots, self.train_samples_per_class)
        self.test_batch = self.prepare_batch(self.class_ids, self.class_rots, self.test_samples_per_class)
        # self.get_train_batch()
        # self.get_test_batch()

    def prepare_batch(self, class_ids, class_rots, samples_per_class_num):
        res_lst = []
        rand_rots = np.zeros(samples_per_class_num * len(class_ids))
        ind = 0
        for i in range(0, len(class_ids)):
            coll, ids = self.loader.get_image_collection(class_ids[i], N = samples_per_class_num, do_transform=False)
            res_lst.append(coll)
            for ti in range(0, samples_per_class_num):
                rand_rots[ind] = class_rots[i]
                ind += 1
        return res_lst, rand_rots

    def get_train_batch(self):
        # self.train_batch = self.prepare_batch(self.class_ids, self.train_samples_per_class)
        # print 'classes: '
        # for i in range(0, len(self.class_ids)):
        #     print self.class_ids[i]
        return self.train_batch

    def get_test_batch(self):
        # self.test_batch = self.prepare_batch(self.class_ids, self.test_samples_per_class)
        return self.test_batch

class Loader:

    def __init__(self, db_path, metadb_path, caffe_root, imside, do_load_stats = False, filterQFile = ''):
        f_in = open(metadb_path, 'r')
        self.group_to_image = []


        for line in f_in:
            parts = line.split(' ')
            ids = []
            for p in parts:
                if (len(p) == 0 or p == '\n'):
                    continue
                ids.append(int(p.split('\n')[0]))

            self.group_to_image.append(ids)

        self.group_num = len(self.group_to_image)
        self.db_path = db_path
        self.env = lmdb.open(db_path, readonly=True)

        self.transformer = caffe.io.Transformer({'data': (1, 3, imside, imside)})
        mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)
        self.transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
        self.transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
        self.transformer.set_raw_scale('data', 1)  # rescale from [0, 1] to [0, 255]
        self.transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
        self.do_load_stats = do_load_stats

        self.get_max_id()

        self.dup_flag = np.ones(self.max_id+1, dtype=np.int32)
        self.do_filter_query = False
        if (len(filterQFile) > 0):
            self.do_filter_query = True
            self.filter_q_file = filterQFile
            for line in (open(filterQFile, 'r')):
                if (len(line) > 1):
                    id = int(line[0:-1])
                    self.dup_flag[id] = 0




    def get_max_id(self):
        self.max_id = -1
        for i in range(0, len(self.group_to_image)):
            if (len(self.group_to_image[i])> 0):
                max_id_in_group = np.max(self.group_to_image[i])
                if (max_id_in_group > self.max_id):
                    self.max_id = max_id_in_group

    def set_group_cnt(self, group_cnt):
        self.group_to_image = self.group_to_image[0:group_cnt]
        self.get_max_id()


    def choose_N_from_list(self, N, perm_ids, w_rep = False):
        # return perm_ids[0:N]
        if (w_rep):
            chosen_ids = []
            while (len(chosen_ids) < N):
                ri = np.random.randint(0, len(perm_ids))
                chosen_ids.append(perm_ids[ri])
            return chosen_ids
        else:
            chosen_ids = []
            while (len(chosen_ids) < N):
                ri = np.random.randint(0, len(perm_ids))
                chosen_ids.append(perm_ids.pop(ri))
            return chosen_ids

    def open_transaction(self):
        try:
            self.txn = self.env.begin()
        except Exception as inst:
            print str(inst)
            self.env = lmdb.open(self.db_path, readonly=True)
            self.txn = self.env.begin()

    def get_image_collection(self, group_id, does_belong = True, N = -1, do_transform=True, w_rep = False):

        chosen_ids = []
        perm_ids = []

        if (group_id >= len(self.group_to_image)):
            return [], []

        if (N<0):
            if (does_belong):
                chosen_ids = []

                if (self.do_filter_query):
                    for j in range(0, len(self.group_to_image[group_id])):
                        id = self.group_to_image[group_id][j]
                        if (self.dup_flag[id] == 1):
                            chosen_ids.append(id)
                else:
                    chosen_ids = list(self.group_to_image[group_id])

            else:
                print 'Loader Error: No support of N<0 from does not belong'
                exit()
                pass

        if (N > 0):
            if (does_belong):
                perm_ids = []
                if (self.do_filter_query):
                    for j in range(0, len(self.group_to_image[group_id])):
                        id = self.group_to_image[group_id][j]
                        if (self.dup_flag[id] == 1):
                            perm_ids.append(id)
                else:
                    perm_ids = list(self.group_to_image[group_id])

                if (len(perm_ids) > N):
                    chosen_ids = self.choose_N_from_list(N, perm_ids, w_rep)
                else:
                    chosen_ids = perm_ids
            if (not does_belong):
                if (self.do_load_stats):
                    negs = self.neg_chosen[group_id]
                    for i in range(0, len(negs)):
                        gi = int(negs[i][0])
                        ii = int(negs[i][1])
                        ind = self.group_to_image[gi][ii]
                        perm_ids.append(ind)
                    chosen_ids = self.choose_N_from_list(N, perm_ids)
                else:
                    chosen_ids = []
                    while (len(chosen_ids) < N):
                        rand_id = np.random.randint(0, self.max_id)
                        lb = self.group_to_image[group_id][0]
                        ub = self.group_to_image[group_id][-1]
                        if (rand_id >= lb and rand_id <= ub):
                            continue

                        chosen_ids.append(rand_id)

        s1 = time.time()
        try:
            txn = self.env.begin()
        except Exception as inst:
            print str(inst)
            self.env = lmdb.open(self.db_path, readonly=True)
            txn = self.env.begin()

        res_img_list = []
        for id in chosen_ids:
            arr = self.get_img_by_id(id, txn, do_transform)
            res_img_list.append(arr)

        e1 = time.time()
        # print 'ldr db ' + str(e1-s1)

        return res_img_list, chosen_ids

    def get_img_by_id(self, id, txn, do_transform = True):
        val = txn.get(str(id))
        datum = caffe.proto.caffe_pb2.Datum.FromString(val)
        arr = caffe.io.datum_to_array(datum)
        if (do_transform):
            arr = self.transformer.preprocess('data', arr)
        return arr
