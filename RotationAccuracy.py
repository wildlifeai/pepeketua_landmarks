import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class RotationAccuracy(keras.metrics.Metric):
    ROTATION_ACCURACY_FORMAT = 'angle_accuracy_radius_{0}'
    ROTATION_OUTSIDE_AVERAGE_DISTANCE_FORMAT = 'angle_outside_radius_{0}_distance'
    def __init__(self, name = None, accuracy = True, angle_radius = 0, **kwargs):
        if name == None:
            name = self.ROTATION_ACCURACY_FORMAT.format(angle_radius) if accuracy else self.ROTATION_OUTSIDE_AVERAGE_DISTANCE_FORMAT.format(angle_radius)
        super(RotationAccuracy, self).__init__(name = name, **kwargs)
        self.accuracy = accuracy
        self.angle_radius = angle_radius
        self.num_points = self.add_weight(name = 'num_points', initializer = 'zeros')
        self.total_angle_distance_outside = self.add_weight(name = 'total_angle_distance_outside', initializer = 'zeros')
        self.num_points_outside = self.add_weight(name = 'num_points_outside', initializer = 'zeros')


    def update_state(self, y_true, y_pred, sample_weight = None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        theta_true = tf.atan2(y_true[: ,0], y_true[:, 1])
        theta_pred = tf.atan2(y_pred[:, 0], y_pred[:, 1])
        diffs_1 = tf.abs(theta_true - theta_pred)
        diffs_2 = tf.abs(theta_true - theta_pred + (2 * np.pi))
        diffs_3 = tf.abs(theta_true - theta_pred - (2 * np.pi))
        diffs = tf.minimum(tf.minimum(diffs_1, diffs_2), diffs_3)
        diffs = diffs * 360 / (2 * np.pi)
        self.num_points.assign_add(tf.cast(tf.size(diffs), tf.float32))

        outside_points = diffs > self.angle_radius
        self.num_points_outside.assign_add(tf.reduce_sum(tf.cast(outside_points, tf.float32)))

        outside_diffs = diffs[outside_points]
        self.total_angle_distance_outside.assign_add(tf.reduce_sum(outside_diffs))

    def result(self):
        if self.accuracy:
            return 1 - self.num_points_outside / self.num_points
        if self.num_points_outside == 0:
            return 0.0
        return self.total_angle_distance_outside / self.num_points_outside

    def reset_states(self):
        self.num_points.assign(0)
        self.num_points_outside.assign(0)
        self.total_angle_distance_outside.assign(0)