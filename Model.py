from complex_fc_cpu import FullyConnected
from PowerLayer import Power
from LossLayer import MSE
from MaxPooling import MaxPooling
from EGC import EGC

class Model:
    def __init__(self, num_beams, num_ant, batch_size, mode='orig', accum=False):
        # layers
        self.ComplexFC = FullyConnected(num_beams, num_ant, batch_size, mode, accum)
        self.Power = Power(num_beams)
        self.MaxPool = MaxPooling(num_beams)
        self.EGC = EGC(num_ant)
        self.Loss = MSE(batch_size)

        # codebook and gradient
        self.batch_size = batch_size
            # 码字即为FC层的权重
        self.codebook = self.ComplexFC.thetas
        self.grad = self.ComplexFC.grad

    def forward(self, h, val_mode=False, val_size=100):
        Z = self.ComplexFC.forward(h, val_mode)
        Q = self.Power.forward(Z, val_mode)
        cb_power = self.MaxPool.forward(Q, val_mode)
        egc_power = self.EGC.forward(h)
        loss = self.Loss.forward(egc_power, cb_power, val_mode, val_size)
        return loss

    def backward(self):
        dL_dP = self.Loss.backward()
        dL_dQ = self.MaxPool.backward(dL_dP)
        dL_dZ = self.Power.backward(dL_dQ)
        dL_dW = self.ComplexFC.backward(dL_dZ)
        self.grad = dL_dW
        return dL_dW

    def update(self, lr=0.1):
        self.codebook = self.ComplexFC.update(lr=lr)
        return self.codebook
