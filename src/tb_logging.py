import pickle
import gzip
from torch.utils.tensorboard import SummaryWriter
import copy
import os

class Logger:
    def __init__(self, output_path):
        self.data = {}
        self.output_path = output_path
        self.context = ""

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]

    def add_object(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def save(self):
        path = os.path.join(self.output_path, 'log')
        pickle.dump(self.data, gzip.open(path, 'wb'))


class TensorboardLogger(Logger):
    def __init__(self, output_path):
        self.data = {}
        self.context = ""
        self.output_path = output_path
        tb_dir = os.path.join(output_path, 'tb_log')
        self.writer = SummaryWriter(log_dir=tb_dir, comment=f"")

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, idx=None, use_context=True):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]

        if idx is None:
            idx = len(self.data[key])
        self.writer.add_scalar(key, value, idx)

    def add_object(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def save(self):
        path = os.path.join(self.output_path, 'logs')
        pickle.dump(self.data, gzip.open(path, 'wb'))

    def draw_histogram(self, key, value, step):
        self.writer.add_histogram(key, value, step, bins='auto')
