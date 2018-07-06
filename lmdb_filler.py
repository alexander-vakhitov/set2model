import os, lmdb
import numpy as np, cv2
import caffe

def read_wnids(syn_list_path):
    f_in = open(syn_list_path, 'r')
    syn_lst = []
    for line in f_in:
        if (len(line) == 0):
            continue
        syn_code = line.split('\n')[0]
        syn_lst.append(syn_code)
    return syn_lst

def check_synset(class_fldr, orig_fldr):
    num_tar = 0
    if (os.path.exists(orig_fldr)):
        num_tar = len(os.listdir(orig_fldr))
    num_web = 0
    s0 = 0
    s1 = 0
    web_fldr = class_fldr + '/web'
    if (os.path.exists(web_fldr)):
        fl = os.listdir(web_fldr)
        num_web = len(fl)
        if (num_web > 1):
            s0 = os.path.getsize(web_fldr + '/' + fl[0])
            s1 = os.path.getsize(web_fldr + '/' + fl[1])
    return num_tar, num_web, s0, s1


def form_square_image(img):
    max_side = np.max(img.shape)
    img_new = np.zeros((max_side, max_side), dtype=np.uint8)
    ud_len = img_new.shape[0]-img.shape[0]
    lr_len = img_new.shape[1]-img.shape[1]
    if (np.mod(ud_len, 2) == 0):
        u = ud_len/2
        d = ud_len/2
    else:
        u = (ud_len-1)/2
        d = (ud_len+1)/2
    if (np.mod(lr_len, 2) == 0):
        l = lr_len/2
        r = lr_len/2
    else:
        l = (lr_len-1)/2
        r = (lr_len+1)/2
    img_new = cv2.copyMakeBorder(img, u, d, l, r, cv2.BORDER_REPLICATE)
    return img_new

def add_image_folder_to_db(tmp_path, img_paths, key_counter, key_lst, group_id, txn, resize_shape = (227, 227), is_grayscale = False):

    for fpath in img_paths:
        img = np.zeros(0)
        if (is_grayscale):
            img = cv2.imread(tmp_path + '/' + fpath, 0)
        else:
            img = cv2.imread(tmp_path + '/' + fpath)

        if (img is None or img.shape[0] == 0):
            continue
        if (resize_shape[0] > 0):
            sqim = form_square_image(img)
            img = cv2.resize(sqim, resize_shape)

        datum = caffe.proto.caffe_pb2.Datum()
        if (is_grayscale):
            datum.channels = 1
            datum.height = img.shape[0]
            datum.width = img.shape[1]
        else:
            datum.channels = img.shape[0]
            datum.height = img.shape[1]
            datum.width = img.shape[2]
        datum.data = img.tostring()
        datum.label = group_id
        txn.put(str(key_counter), datum.SerializeToString())
        key_lst.append(key_counter)
        key_counter += 1
    return key_counter

def fill_db(N, env_q, env_t, group_id, group_to_key_lst_q, group_to_key_lst_t, imgnet_path, key_counter_q,
            key_counter_t, txn_q, txn_t, lst_fname):
    synlst_path = imgnet_path + '/' + lst_fname
    synlst = read_wnids(synlst_path)
    if (N > 0):
        synlst = synlst[0:N]
    for wnid in synlst:
        class_fldr = imgnet_path + '/imgs/' + wnid
        if (not os.path.exists(class_fldr)):
            continue

        tar_path = class_fldr + '/imgnet/orig.tar'
        orig_fldr = class_fldr + '/imgnet'
        img_paths = []
        tmp_path = orig_fldr
        img_paths = os.listdir(orig_fldr)

        t_imgs_path = class_fldr + '/web'
        if (len(t_imgs_path) == 0 or len(img_paths) == 0):
            continue

        num_tar, num_web, s0, s1 = check_synset(class_fldr, orig_fldr)
        if (num_web == 0 or num_tar==0 or (s0+s1==0)):
            break

        key_lst_t = []
        print 'adding to q group.. ' + str(group_id)
        key_counter_t = add_image_folder_to_db(tmp_path, img_paths, key_counter_t, key_lst_t, group_id, txn_t)


        t_img_paths = os.listdir(t_imgs_path)

        key_lst_q = []
        print 'adding to t group.. ' + str(group_id)
        key_counter_q = add_image_folder_to_db(t_imgs_path, t_img_paths, key_counter_q, key_lst_q, group_id, txn_q)
        print 'added group ' + str(group_id)
        group_to_key_lst_q.append(key_lst_q)
        group_to_key_lst_t.append(key_lst_t)
        group_id += 1

        txn_q.commit()
        txn_t.commit()

        txn_q = env_q.begin(write=True)
        txn_t = env_t.begin(write=True)

    return group_id, group_to_key_lst_q, group_to_key_lst_t, key_counter_q, key_counter_t


def read_group_from_db(db_path, chosen_ids, isq=True):
    env = lmdb.open(db_path+'/testq', readonly=True)
    if (not isq):
        env = lmdb.open(db_path + '/testt', readonly=True)
    txn = env.begin()
    res_img_list = []
    for id in chosen_ids:
        val = txn.get(str(id))
        datum = caffe.proto.caffe_pb2.Datum.FromString(val)
        arr = caffe.io.datum_to_array(datum)

        res_img_list.append(arr)

    return res_img_list






def check_download_folder(fldr, lst_fname):
    syn_lst_path = fldr + lst_fname
    save_path = fldr + '/imgs'

    syn_lst = read_wnids(syn_lst_path)
    ind = 0
    while (ind < len(syn_lst)):
        class_fldr = save_path+'/'+syn_lst[ind]
        orig_fldr = class_fldr+'/imgnet'

        num_tar, num_web, s0, s1 = check_synset(class_fldr, orig_fldr)
        if (num_web == 0 or num_tar==0 or (s0+s1==0)):
            break
        ind += 1

    return ind


class Filler:
    def init_dual_lmdb(self, db_dir):
        self.lmdb_test_db_path_q = db_dir + '/' + 'testq'
        self.test_meta_db_path_q = db_dir + '/' + 'meta_test_q'
        self.lmdb_test_db_path_t = db_dir + '/' + 'testt'
        self.test_meta_db_path_t = db_dir + '/' + 'meta_test_t'

        group_to_key_lst_q, key_counter_q, group_id_q = self.read_meta_db(self.test_meta_db_path_q)
        self.group_to_key_lst_q = group_to_key_lst_q
        self.key_counter_q = key_counter_q

        group_to_key_lst_t, key_counter_t, group_id_t = self.read_meta_db(self.test_meta_db_path_t)
        self.group_to_key_lst_t = group_to_key_lst_t
        self.key_counter_t = key_counter_t

        if (group_id_q != group_id_t):
            print 'Error in Loader: test and query have diff num of groups'
        else:
            self.group_id = group_id_q

    def init_single_lmdb(self, db_dir):
        self.db_dir = db_dir
        self.lmdb_test_db_path_all = db_dir + '/' + 'test_all'
        self.test_meta_db_path_all = db_dir + '/' + 'meta_test_all'

        group_to_key_lst_all, key_counter_all, group_id_all = self.read_meta_db(self.test_meta_db_path_all)
        self.group_to_key_lst_all = group_to_key_lst_all
        self.key_counter_all = key_counter_all
        self.group_id = group_id_all

    def __init__(self, db_dir, init_mode =1 ):
        self.init_mode = init_mode
        if (init_mode == 1):
            self.init_dual_lmdb(db_dir)
        if (init_mode == 2):
            self.init_single_lmdb(db_dir)

    def cut_base(self, max_group_id):
        if (self.init_mode == 1):
            self.group_to_key_lst_t = self.group_to_key_lst_t[0:max_group_id+1]
            self.group_to_key_lst_q = self.group_to_key_lst_q[0:max_group_id + 1]
            self.key_counter_q = np.max(self.group_to_key_lst_q[-1]) + 1
            self.key_counter_t = np.max(self.group_to_key_lst_t[-1]) + 1

    def read_meta_db(self, meta_db_path_q):
        group_to_key_lst_q = []
        key_cntr = 0
        group_id = 0

        if (not os.path.exists(meta_db_path_q)):
            return group_to_key_lst_q, key_cntr, group_id

        f_in = open(meta_db_path_q, 'r')

        for line in f_in:
            parts = line.split(' ')
            if (len(parts) < 1):
                continue
            gl = []
            for p in parts:
                if (p != '\n'):
                    gl.append(int(p))
            if (len(gl) > 0):
                key_cntr = np.max(gl) + 1
                group_to_key_lst_q.append(gl)
                group_id += 1

        return group_to_key_lst_q, key_cntr, group_id

    def output_meta_db(self, meta_db_path_q, group_to_key_lst_q):
        f_out = open(meta_db_path_q, 'w')
        for gl in group_to_key_lst_q:
            for kid in gl:
                f_out.write(str(kid) + ' ')
            f_out.write('\n')

    def load_base_with_queries(self, imgnet_path, N=0, lst_fname='train.txt'):
        if not os.path.isdir(self.lmdb_test_db_path_q):
            os.makedirs(self.lmdb_test_db_path_q)
        env_q = lmdb.open(self.lmdb_test_db_path_q, map_size=1e12, max_dbs=1)
        txn_q = env_q.begin(write=True)
        if not os.path.isdir(self.lmdb_test_db_path_t):
            os.makedirs(self.lmdb_test_db_path_t)
        env_t = lmdb.open(self.lmdb_test_db_path_t, map_size=1e12, max_dbs=1)
        txn_t = env_t.begin(write=True)

        self.group_id, self.group_to_key_lst_q, self.group_to_key_lst_t, \
        self.key_counter_q, self.key_counter_t = fill_db(N, env_q, env_t, self.group_id, self.group_to_key_lst_q, self.group_to_key_lst_t, imgnet_path, self.key_counter_q,
                self.key_counter_t, txn_q, txn_t, lst_fname)

        self.output_meta_db(self.test_meta_db_path_q, self.group_to_key_lst_q)
        self.output_meta_db(self.test_meta_db_path_t, self.group_to_key_lst_t)


def prepare_lmdb_database(db_folder, image_folder, db_label, lst_fname):
    db_dir = db_folder + '/' + db_label + '/'
    filler = Filler(db_dir)
    N = check_download_folder(image_folder, lst_fname)
    filler.load_base_with_queries(image_folder, N, lst_fname)