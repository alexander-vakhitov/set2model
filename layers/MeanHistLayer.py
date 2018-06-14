import GaussHistLossLayer as ghl
import caffe
import theano.tensor as T
from theano import function

class MeanHistLossLayer(caffe.Layer, ghl.HistLossLayer):


    def calc_mean_ll_descs(self, X, Y):
        meanvec = T.mean(X, 0)
        D = -T.sqrt(T.sum((Y - T.tile(meanvec, (Y.shape[0], 1)))**2, 1))
        return D

    def calc_mean_hist_loss(self, X, Yp, Yn):
        pp = self.calc_mean_ll_descs(X, Yp)
        pn = self.calc_mean_ll_descs(X, Yn)
        L, hmax, hmin, hn, hp = self.calc_hist_loss_vector(pn, pp)
        return L, hmax, hmin, hn, hp

    def initialize_mean_fun(self):
        X = T.fmatrix('X')
        Yp = T.fmatrix('Yp')
        Yn = T.fmatrix('Yn')
        L, hmax, hmin, hn, hp = self.calc_mean_hist_loss(X, Yp, Yn)
        self.f_mean = function([X, Yp, Yn], L, allow_input_downcast=True)
        dL = T.grad(L, [X, Yp, Yn])
        self.df_mean = function([X, Yp, Yn], dL, allow_input_downcast=True)

    def setup(self, bottom, top):
        self.sample_size = 50
        self.setup_fields()
        self.initialize_mean_fun()
        self.scaling_coeff = 0.00005

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[0] = 0
        self.parse_input(bottom)
        Lval = self.f_mean(self.query_descs, self.pos_descs, self.neg_descs)
        top[0].data[0] = self.scaling_coeff*Lval


    def backward(self, top, propagate_down, bottom):
        self.parse_input(bottom)
        dL = self.df_mean(self.query_descs, self.pos_descs, self.neg_descs)
        if (propagate_down[0]):
            bottom[0].diff[self.q_bnds[0]:self.q_bnds[1], :] = self.scaling_coeff*dL[0][...]
            bottom[0].diff[self.pos_bnds[0]:self.pos_bnds[1], :] = self.scaling_coeff*dL[1][...]
            bottom[0].diff[self.neg_bnds[0]:self.neg_bnds[1], :] = self.scaling_coeff*dL[2][...]
