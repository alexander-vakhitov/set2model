import caffe, numpy as np
import Debugger as dbg

class ActNormLayer(caffe.Layer):


    def setup(self, bottom, top):

        pass

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1])


    def forward(self, bottom, top):

        # dbg.log_forward('act_norm', bottom)

        self.norms = []
        for j in range(0, bottom[0].data.shape[0]):
            self.norms.append(np.linalg.norm(bottom[0].data[j, :].reshape(-1)) + 1e-5)
            top[0].data[j, :] = bottom[0].data[j, :].reshape(bottom[0].data.shape[1]) / self.norms[j]




    def backward(self, top, propagate_down, bottom):
        bottom_desc_shape = bottom[0].shape[1:]
        for di in range(0, top[0].diff.shape[0]):
            diff_val = 1/self.norms[di]*(top[0].diff[di,:] - top[0].data[di,:]*(np.dot(top[0].diff[di,:], top[0].data[di,:])))
            bottom[0].diff[di, :] = diff_val.reshape(bottom_desc_shape)