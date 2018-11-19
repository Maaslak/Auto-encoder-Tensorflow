from Tools.scripts import import_diagnostics
from keras.datasets import cifar10, mnist
from auto_encoder import AutoEncoder




salt_ratio = .05
pepper_ratio = .15

encoder = AutoEncoder.AutoEncoder(data_set=cifar10)
encoder.train()


