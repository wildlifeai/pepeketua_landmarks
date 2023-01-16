import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class MultiGen(keras.utils.Sequence):
    """docstring for MultiGen"""

    def __init__(self, cl, num_instances, *args, **kargs):
        super(MultiGen, self).__init__()
        self.num_instances = num_instances
        self.cl_instances = []
        self.batch_size = kargs["batch_size"]
        self.training = kargs["training"]
        self.cl_batch_size = math.ceil(self.batch_size / num_instances)
        kargs["batch_size"] = self.cl_batch_size
        for i in range(num_instances):
            self.cl_instances.append(cl(*args, **kargs))

        self.on_epoch_end()

    def __getitem__(self, index):
        items = []
        for cl_in in self.cl_instances:
            items.append(cl_in.__getitem__(index))

        num_inner_items = len(items[0])
        items_holder = [[] for i in range(num_inner_items)]
        for i in range(num_inner_items):
            for item in items:
                items_holder[i].append(item[i])

        ret_items = []
        for item in items_holder:
            ret_items.append(np.concatenate(item, axis=0))

        if self.training:
            return ret_items[0], ret_items[1]
        return ret_items[0]

    def __len__(self):
        return self.cl_instances[0].__len__()

    def on_epoch_end(self):
        for cl_in in self.cl_instances:
            cl_in.on_epoch_end()


if __name__ == "__main__":
    main()
