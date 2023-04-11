import numpy as np

class EGC:
    def __init__(self, num_ant):
        self.num_ant = num_ant
        self.egc_power = 0

    def forward(self, channel):
        channel_real_square = np.power(channel[:self.num_ant], 2)
        channel_imag_square = np.power(channel[self.num_ant:], 2)
        self.egc_power = np.power(np.sum(np.sqrt(channel_real_square + channel_imag_square)), 2) / self.num_ant
        return self.egc_power
        