import caffe, numpy as np
import theano.tensor as T
from theano import function
import yaml
import time
import sklearn.mixture as mixture


class HistLossLayer:
    def __init__(self):
        pass

    def parse_input(self, bottom):
        self.all_labels = np.zeros_like(bottom[1].data)
        self.all_labels[...] = bottom[1].data[...]
        self.group_id = bottom[2].data[0]
        p_lb = 0
        i = 0
        while (self.all_labels[i] == 1):
            i += 1
        p_ub = i
        while (self.all_labels[i] != 0):
            i += 1
        n_lb = i
        while (self.all_labels[i] == 0):
            i += 1
        n_ub = i
        while (self.all_labels[i] != 2):
            i += 1
        q_lb = i
        while (i < len(self.all_labels) and self.all_labels[i] == 2):
            i += 1
        q_ub = i
        self.pos_bnds = (p_lb, p_ub)
        self.neg_bnds = (n_lb, n_ub)
        self.q_bnds = (q_lb, q_ub)
        self.query_descs = bottom[0].data[self.q_bnds[0]:self.q_bnds[1], :]
        self.pos_descs = bottom[0].data[self.pos_bnds[0]:self.pos_bnds[1], :]
        self.neg_descs = bottom[0].data[self.neg_bnds[0]:self.neg_bnds[1], :]


class GaussHistLossLayer(caffe.Layer, HistLossLayer):

    def initialize_theano_fun(self):
        X = T.fmatrix('X')
        Yp = T.fmatrix('Yp')
        Yn = T.fmatrix('Yn')

        L, hp, hn, hmax, hmin = self.calc_hist_loss_gauss_vector(X, Yp, Yn)
        self.f = function([X, Yp, Yn], [L, hp, hn, hmax, hmin], allow_input_downcast=True)

        gf = T.grad(L, [X, Yp, Yn])
        self.df = function([X, Yp, Yn], gf, allow_input_downcast=True)

        return

    def setup(self, bottom, top):
        self.params = yaml.load(self.param_str)
        if ('sample_size' in self.params.keys()):
            self.sample_size = self.params['sample_size']
        else:
            self.sample_size = 50

        self.bin_num = 100
        self.sample_num = self.sample_size
        self.hmin = 0.0
        self.hmax = 1.0
        self.min_cov = 1e-6
        self.reg_coef = 1e-5

        self.initialize_theano_fun()

        self.scaling_coeff = 0.00005
        self.cnt = 0

    def reshape(self, bottom, top):
        top[0].reshape(1)


    def forward(self, bottom, top):
        self.parse_input(bottom)
        [Lval, hp, hn, hmax, hmin] = self.f(self.query_descs, self.pos_descs, self.neg_descs)
        top[0].data[0] = self.scaling_coeff*Lval


    def backward(self, top, propagate_down, bottom):
        s1 = time.time()
        self.parse_input(bottom)

        dL = self.df(self.query_descs, self.pos_descs, self.neg_descs)

        if (propagate_down[0]):
            bottom[0].diff[self.q_bnds[0]:self.q_bnds[1], :] = self.scaling_coeff*dL[0]
            bottom[0].diff[self.pos_bnds[0]:self.pos_bnds[1], :] = self.scaling_coeff*dL[1]
            bottom[0].diff[self.neg_bnds[0]:self.neg_bnds[1], :] = self.scaling_coeff*dL[2]

        bottom[1].diff[...] = np.zeros_like(bottom[1].diff)
        bottom[2].diff[...] = np.zeros_like(bottom[2].diff)
        bottom[3].diff[...] = np.zeros_like(bottom[3].diff)
        self.cnt += 1


class GMMContainer():

    def __init__(self, means, covars, weights):
        self.means = means
        self.covars = covars
        self.weights = weights
        self.cnt = 0


class GMMHistLossLayer(caffe.Layer, HistLossLayer):

    def initialize_calc_ll_gmm_hist_fun(self):
        meansvec = T.dvector('means')
        covarsvec = T.dvector('covars')
        weights = T.dvector('weights')
        gm_num = weights.shape[0]
        means = T.reshape(meansvec, (gm_num, meansvec.shape[0] / gm_num))
        covars = T.reshape(covarsvec, (gm_num, meansvec.shape[0] / gm_num))
        Yp = T.dmatrix('Yp')
        Yn = T.dmatrix('Yn')
        p_p,r_p,p_p_m = self.calc_ll_gmm(Yp, means, covars, weights)
        p_n,r_n,p_n_m = self.calc_ll_gmm(Yn, means, covars, weights)

        L, hmax, hmin, hn, hp = self.calc_hist_loss_vector(p_n, p_p)
        dL = T.jacobian(L, [meansvec, covarsvec, weights, Yp, Yn])
        self.gmmhist_df = function([meansvec, covarsvec, weights, Yp, Yn], dL, allow_input_downcast=True)
        self.gmmhist_f = function([meansvec, covarsvec, weights, Yp, Yn], [L, hmax, hmin, hn, hp], allow_input_downcast=True)

    def setup(self, bottom, top):
        self.params = yaml.load(self.param_str)
        self.gm_num = self.params['gm_num']
        if ('sample_size' in self.params.keys()):
            self.sample_size = self.params['sample_size']
        else:
            self.sample_size = 50

        self.bin_num = 100
        self.sample_num = self.sample_size
        self.hmin = 0.0
        self.hmax = 1.0
        self.min_cov = 1e-6
        self.reg_coef = 1e-5
        self.initialize_calc_ll_gmm_hist_fun()
        self.gmm_dict = {}
        self.scaling_coeff = 0.00005

    def reshape(self, bottom, top):
        top[0].reshape(1)


    def build_gmm(self, X, n_it = 1000, min_cov = 0.01):
        gmm = mixture.GMM(covariance_type='diag', init_params='wmc', min_covar=min_cov,
                    n_components=self.gm_num, n_init=1, n_iter=n_it, params='wmc',
                    random_state=None)
        gmm.fit(X)


        return np.copy(gmm.means_), np.copy(gmm.covars_), np.copy(gmm.weights_), gmm.score(X)

    def refine_gmm(self, X, means, covs, weights, min_covar = 0.000001, n_iter = 1000):
        gm_num = weights.shape[0]
        gmm = mixture.GMM(covariance_type='diag', init_params='', min_covar=min_covar,
                    n_components=gm_num, n_init=1, n_iter=n_iter, params='wmc',
                    random_state=None)
        gmm.means_ = means
        gmm.covars_ = covs
        gmm.weights_ = weights
        gmm.fit(X)
        return np.copy(gmm.means_), np.copy(gmm.covars_), np.copy(gmm.weights_)



    def forward(self, bottom, top):
        self.parse_input(bottom)
        gmm_data = []
        if self.group_id in self.gmm_dict:
            gmm_data = self.gmm_dict[self.group_id]
        else:
            if (self.gm_num > 0):
                [means, covars, weights, score] = self.build_gmm(self.query_descs)
                gmm_data = GMMContainer(means, covars, weights)
            else:
                [means, covars, weights, score] = self.build_adagmm(self.query_descs)
                gmm_data = GMMContainer(means, covars, weights)

        [means, covars, weights] = self.refine_gmm(self.query_descs, gmm_data.means, gmm_data.covars, gmm_data.weights)
        gmm_data.means = means
        gmm_data.covars = covars
        gmm_data.weights = weights
        gmm_data.cnt += 1
        self.gmm_dict[self.group_id] = gmm_data
        self.meansvec = np.reshape(means, (np.prod(means.shape)))
        self.covsvec = np.reshape(covars, (np.prod(covars.shape)))
        self.weights = weights
        L, hmax, hmin, hn, hp = self.gmmhist_f(self.meansvec, self.covsvec, weights, self.pos_descs, self.neg_descs)

        top[0].data[0] = self.scaling_coeff*L


    def calc_gmm_probs_dif(self, X, Yp, Yn, meansvec, covarsvec, weights):
        Xvec = np.reshape(X, np.prod(X.shape))
        dX = self.solve_lin_sys_for_gmm(Xvec, meansvec, covarsvec, weights)
        df = self.gmmhist_df(meansvec, covarsvec, weights, Yp, Yn)
        df_vec = np.concatenate((df[0], df[1], df[2]))
        dXf = (df_vec).dot(dX[0:dX.shape[0]-1, :])
        return df[3], df[4], dXf

    def solve_lin_sys_for_gmm(self, Xvec, meansvec, covarsvec, weights):
        gm_num = len(weights)
        n_dim = len(meansvec)/len(weights)
        n_samples = len(Xvec)/n_dim
        lam = n_samples
        hm = self.gmm_hm(Xvec, meansvec, covarsvec, weights, lam)
        hc = self.gmm_hc(Xvec, meansvec, covarsvec, weights, lam)
        hw = self.gmm_hw(Xvec, meansvec, covarsvec, weights, lam)
        f0 = time.time()

        mean_row = np.concatenate((hm[1], hm[2], hm[3], np.zeros((len(meansvec), 1))), axis=1)
        cov_row = np.concatenate((hc[1], hc[2], hc[3], np.zeros((len(meansvec), 1))), axis=1)
        weight_row = np.concatenate((hw[1], hw[2], hw[3], np.reshape(hw[4], (gm_num, 1))), axis=1)
        lambda_row = np.concatenate(
            (np.zeros((1, len(meansvec))),
             np.zeros((1, len(meansvec))),
             np.reshape(hw[4], (1, gm_num)),
             np.zeros((1, 1))), axis=1)

        M = np.concatenate((mean_row, cov_row, weight_row, lambda_row))
        N = np.concatenate((-hm[0], -hc[0], -hw[0], np.zeros((1, hw[0].shape[1]))))

        par_dim = gm_num * n_dim
        a = np.diag(M[0:par_dim, 0:par_dim])
        A = np.diag(a)
        b = np.diag(M[0:par_dim, par_dim:2 * par_dim])
        B = np.diag(b)
        c = np.diag(M[par_dim:2 * par_dim, par_dim:2 * par_dim])
        C = np.diag(c)

        dX = []

        s1 = 0
        fs1 = 0

        if (np.linalg.norm(A - M[0:par_dim, 0:par_dim]) < 1e-15 and
            np.linalg.norm(B - M[0:par_dim, par_dim:2*par_dim]) < 1e-15 and
            np.linalg.norm(C - M[par_dim:2*par_dim, par_dim:2 * par_dim]) < 1e-15):

            D = M[2 * par_dim:, 2 * par_dim:]

            if (np.linalg.matrix_rank(A) < A.shape[0]):
                a = a + np.ones(a.shape[0])*self.reg_coef
            if (np.linalg.matrix_rank(C) < C.shape[0]):
                c = c + np.ones(a.shape[0]) * self.reg_coef
            if (np.linalg.matrix_rank(D) < D.shape[0]):
                D = D + np.eye(D.shape[0]) * self.reg_coef

            Di = np.linalg.inv(D)
            e = 1 / (a - b / c * b)
            f = -e * b / c
            h = (np.ones(a.shape[0]) - f * b) / c

            dX = np.zeros(N.shape)
            n_samples = 10
            for i in range(0, n_samples):
                N1 = N[:, i * n_dim:(i + 1) * n_dim]
                for gi in range(0, gm_num):
                    n_mu_gi = np.diag(N1[gi * n_dim:(gi + 1) * n_dim, 0:n_dim])
                    e_gi = e[gi * n_dim:(gi + 1) * n_dim]
                    n_cov_gi = np.diag(
                        N1[n_dim * gm_num + gi * n_dim:n_dim * gm_num + (gi + 1) * n_dim, 0:n_dim])
                    f_gi = f[gi * n_dim:(gi + 1) * n_dim]
                    h_gi = h[gi * n_dim:(gi + 1) * n_dim]
                    dX[gi * n_dim: (gi + 1) * n_dim, i * n_dim:(i + 1) * n_dim] = np.diag(
                        e_gi * n_mu_gi + f_gi * n_cov_gi)
                    dX[n_dim * gm_num + gi * n_dim: n_dim * gm_num + (gi + 1) * n_dim,
                    i * n_dim: (i + 1) * n_dim] = np.diag(f_gi * n_mu_gi + h_gi * n_cov_gi)

            dX[n_dim * 2 * gm_num:, :] = Di.dot(N[n_dim * 2 * gm_num:, :])

        else:
            M = M + self.reg_coef * np.eye(M.shape[0])
            dX = np.linalg.solve(M, N)

        return dX


    def backward(self, top, propagate_down, bottom):

        self.parse_input(bottom)
        dYp, dYn, dX = self.calc_gmm_probs_dif(self.query_descs, self.pos_descs, self.neg_descs, self.meansvec, self.covsvec, self.weights)
        dX = np.reshape(dX, self.query_descs.shape)
        if (propagate_down[0]):
            bottom[0].diff[self.q_bnds[0]:self.q_bnds[1], :] = self.scaling_coeff*dX[...]
            bottom[0].diff[self.pos_bnds[0]:self.pos_bnds[1], :] = self.scaling_coeff*dYp[...]
            bottom[0].diff[self.neg_bnds[0]:self.neg_bnds[1], :] = self.scaling_coeff*dYn[...]

