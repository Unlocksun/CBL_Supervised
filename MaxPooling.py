import numpy as np

class MaxPooling:
    def __init__(self, num_beams):
        self.S = []
        self.num_beams = num_beams
        self.pow_vec = 0

    def forward(self, inputs, val_mode = False):
        if  val_mode:
            return inputs.max()
        else:
            self.pow_vec = inputs
            return self.pow_vec.max()
        
    def backward(self, dl_dpb):
        pos = self.pow_vec.argmax()
        dpb_dp = np.zeros([1, self.num_beams])
        dpb_dp[0, pos] = dl_dpb
        return dpb_dp
