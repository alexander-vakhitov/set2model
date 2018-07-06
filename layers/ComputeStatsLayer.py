import caffe, yaml, os, numpy as np
import sklearn
import sklearn.mixture as mixture
import warnings

class Mean_RM:
    def __init__(self):
        pass

    def set_query(self, descs):
        self.means = np.mean(descs, 0)

    def get_ranks(self, X):
        M = np.tile(self.means, (X.shape[0], 1))
        d = (M-X)**2
        d = np.sum(d, 1)
        d = np.sqrt(d)
        return -d

    def get_query_mode(self):
        return 0


class GMM_RM:
    def __init__(self, given_gm_num = 2, n_it = 1000):
        self.gm_num = given_gm_num
        self.n_it = n_it

    def build_gmm(self, X, n_it = 1000, min_cov = 0.01):
        gmm = mixture.GMM(covariance_type='diag', init_params='wmc', min_covar=min_cov,
                    n_components=self.gm_num, n_init=1, n_iter=n_it, params='wmc',
                    random_state=None)
        gmm.fit(X)
        return np.copy(gmm.means_), np.copy(gmm.covars_), np.copy(gmm.weights_), gmm.score(X)

    def calc_ll_gmm_noth(self, Y, means, covars, weights):
        n_samples, n_dim = Y.shape
        lpr = (-0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                      + np.sum((means ** 2) / covars, 1)
                      - 2 * np.dot(Y, (means / covars).T)
                      + np.dot(Y ** 2, np.transpose(1.0 / covars))) + np.log(weights))

        lpr = np.transpose(lpr, (1,0))
        # Use the max to normalize, as with the log this is what accumulates
        # the less errors
        vmax = np.max(lpr,axis=0)
        out = np.log(np.sum(np.exp(lpr- vmax), axis=0))
        out += vmax
        responsibilities = np.exp(lpr - np.tile(out, (means.shape[0], 1)))

        return out, responsibilities, np.transpose(lpr)

    def set_query(self, descs):
        [means, covars, weights, score] = self.build_gmm(descs, self.n_it)
        self.means = means
        self.covars = covars
        self.weights = weights

    def get_ranks(self, X):
        p_p, r_p, p_p_m = self.calc_ll_gmm_noth(X, self.means, self.covars, self.weights)
        return p_p

    def get_ranks_resps(self, X):
        p_p, r_p, p_p_m = self.calc_ll_gmm_noth(X, self.means, self.covars, self.weights)
        return p_p, r_p

    def get_scores(self, X):
        p_p, r_p, p_p_m = self.calc_ll_gmm_noth(X, self.means, self.covars, self.weights)
        return p_p_m

    def get_query_mode(self):
        return 0


class ComputeStatsLayer(caffe.Layer):

    def check_params(self, params):
        required = ['gm_num']
        for r in required:
            assert r in params.keys(), 'Params must include {}'.format(r)

    def reshape(self, bottom, top):
        # data
        top[0].reshape(1)




    def setup(self, bottom, top):

        params = yaml.load(self.param_str)
        self.gm_num = int(params['gm_num'])
        self.check_params(params)
        top[0].reshape(1)

        self.report_done = False
        self.q_descs = {}
        self.t_descs = {}
        # self.t_path = params['t_path']
        # self.q_path = params['q_path']

        self.fin_loss = 0

    def make_one_query(self, i, q_for_group, ranking_model, t_descs, pos_t_for_group=np.zeros(1)):
        ranking_model.set_query(q_for_group)
        all_ranks = []
        all_labels = []
        tot_len = 0

        if (len(pos_t_for_group.shape) == 1):
            pos_t_for_group = t_descs[i]
        for ti in range(0, len(t_descs)):
            t_for_group = t_descs[ti]
            if (i == ti):
                r = ranking_model.get_ranks(pos_t_for_group)
                all_ranks.append(r)
                tot_len += r.shape[0]
                all_labels.append(np.ones(r.shape[0]))
            else:
                r = ranking_model.get_ranks(t_for_group)
                all_ranks.append(r)
                tot_len += r.shape[0]
                all_labels.append(np.zeros(r.shape[0]))

        return all_labels, all_ranks, tot_len

    def compute_ap(self, all_labels, all_ranks, aps, tot_len):
        y_true = np.zeros(tot_len)
        y_score = np.zeros(tot_len)
        ind = 0
        for ti in range(0, len(all_ranks)):
            r = all_ranks[ti]
            l = all_labels[ti]
            y_true[ind:ind + r.shape[0]] = l[:]
            y_score[ind:ind + r.shape[0]] = r[:]
            ind += r.shape[0]
        if (np.sum(y_true) > 0):
            ap = sklearn.metrics.average_precision_score(y_true, y_score)
            aps.append(ap)

    def compute_results(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        q_descs_lst = []
        t_descs_lst = []
        for gi in range(0, len(self.q_descs)):
            desc_mat = np.concatenate(self.q_descs[gi])
            q_descs_lst.append(desc_mat)

            # dir_path = self.q_path + '/' + str(gi)+'/'
            # if not os.path.exists(dir_path):
            #     os.makedirs(dir_path)
            # np.save(dir_path + '0.npy', desc_mat)
            #
            # desc_mat = np.concatenate(self.t_descs[gi])
            # t_descs_lst.append(desc_mat)
            #
            # dir_path = self.t_path + '/' + str(gi) + '/'
            # if not os.path.exists(dir_path):
            #     os.makedirs(dir_path)
            # np.save(dir_path + '0.npy', desc_mat)

        if self.gm_num>0:
            grm = GMM_RM(self.gm_num)
        else:
            grm = Mean_RM()
        l_max = np.min([len(self.q_descs), len(self.t_descs)])
        aps = []
        for i in range(0, l_max):  # len(q_descs)):
            q_for_group = q_descs_lst[i]
            all_labels, all_ranks, tot_len = self.make_one_query(i, q_for_group, grm, t_descs_lst)
            self.compute_ap(all_labels, all_ranks, aps, tot_len)
        mAP = np.mean(aps)
        print 'mAP = ' + str(mAP)
        self.fin_loss = mAP


    def forward(self, bottom, top):

        top[0].data[0] = 1.0
        group_id = bottom[2].data[0]

        cnt = 0
        for i in range(0, bottom[1].data.shape[0]):
            if (bottom[1].data[i] >= 0):
                cnt += 1

        if group_id < 0:
            if not self.report_done:
                self.compute_results()
                self.report_done = True
            top[0].data[0] = self.fin_loss
            return

        desc_data = np.copy(bottom[0].data[0:cnt, ...])

        # print 'cnt= ' + str(cnt) +' norm data = '+str(np.linalg.norm(desc_data))

        data_type = bottom[3].data[0]

        if data_type == 1:
            #query base
            if not (group_id in self.q_descs):
                self.q_descs[group_id] = []
            self.q_descs[group_id].append(desc_data)
        else:
            if not (group_id in self.t_descs):
                self.t_descs[group_id] = []
            self.t_descs[group_id].append(desc_data)

    def backward(self, top, propagate_down, bottom):
        pass



