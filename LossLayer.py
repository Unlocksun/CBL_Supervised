import numpy as np

class MSE:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.egc_power = 0
        self.cb_power = 0
        self.loss = 0
        self.count = 0

        self.count_val = 0
        self.loss_val = 0       # 验证集的损失
        self.val_first = True

    def forward(self, egc_power, cb_power, val_mode=False, val_size=100):
        if val_mode:
            if  self.val_first: # 开始验证
                self.count_val = 0
                self.loss_val = 0
                self.val_first = False

            if self.count_val < val_size:
                self.loss_val = self.loss_val + np.square(egc_power - cb_power)
                self.count_val += 1
                if self.count_val == val_size:
                    self.loss_val = (1 / val_size) * self.loss_val
                    self.val_first = True
        # 测试集
        else:
            self.egc_power = egc_power
            self.cb_power = cb_power

            if self.count < self.batch_size:
                self.loss = self.loss_val + np.square(egc_power - cb_power)
                self.count += 1
                if self.count == self.batch_size:
                    self.loss = (1 / self.batch_size) * self.loss
            else:
                self.count = 0
                self.loss = 0
                self.loss = self.loss + np.square(egc_power - cb_power)
                self.count += 1

    def backward(self):
        dl_dpb = 2 * (self.cb_power - self.egc_power)
        return dl_dpb

