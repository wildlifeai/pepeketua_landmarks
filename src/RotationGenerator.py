import tensorflow as tf
from tensorflow import keras

import config
from LandMarkDataGenerator import LandMarkDataGenerator


class RotationGenerator(LandMarkDataGenerator):
    """docstring for RotationGenerator"""

    def __init__(self, normalize_rotation=False, *args, **kwargs):
        super(RotationGenerator, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        if not self.training:
            return super().__getitem__(index)

        images, labels = super().__getitem__(index)

        labels = self.create_rotation_label(labels)
        return images, labels

    def create_rotation_label(self, labels):
        x_snout = labels[:, config.x_tip_of_snout_index]
        y_snout = labels[:, config.y_tip_of_snout_index]
        x_vent = labels[:, config.x_vent_index]
        y_vent = labels[:, config.y_vent_index]

        # Oppisite because image (0,0) top left corner
        # so y axis positive is downward
        x_body = x_vent - x_snout
        y_body = y_vent - y_snout

        x, y = keras.utils.normalize([x_body, y_body], axis=0)

        return tf.transpose(tf.stack([x, y]))
