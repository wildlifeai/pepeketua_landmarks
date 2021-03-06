import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class LocalizationPointAccuracy(keras.metrics.Metric):
	"""
	A Metric to evaluate the percentage of points that fall in
	the radius of the ground truth point or the average distance
	of points that don't

	accuracy - whether to compute percent of points that fall 
				in radius or the average distance of points that 
				don't

	radius - the radius from the ground true points
	"""
	LANDMARK_ACCURACY_FORMAT = 'localization_point_accuracy_radius_{0}'
	LANDMARK_OUTSIDE_AVERAGE_DISTANCE_FORMAT = 'localization_outside_radius_{0}_distance'
	def __init__(self, name = None, accuracy = True, radius = 5, **kwargs):
		if name == None:
			name = self.LANDMARK_ACCURACY_FORMAT.format(radius) if accuracy else self.LANDMARK_OUTSIDE_AVERAGE_DISTANCE_FORMAT.format(radius)
		super(LocalizationPointAccuracy, self).__init__(name = name, **kwargs)
		self.point_outside = self.add_weight(name = 'point_out', initializer = 'zeros')
		self.all_points = self.add_weight(name = 'all_points', initializer = 'zeros')
		self.total_outside_distance = self.add_weight(name = 'total_outside_distance', initializer = 'zeros')
		self.point_radius = radius
		self.accuracy = accuracy

	def update_state(self, y_true, y_pred, sample_weight = None):
		y_true = tf.cast(y_true, tf.float32)
		y_pred = tf.cast(y_pred, tf.float32)
		diffs = tf.square(tf.subtract(y_true, y_pred))
		if len(tf.shape(diffs)) == 1:
			diffs = tf.reduce_sum(tf.reshape(diffs, (1, tf.cast(len(diffs) / 2, tf.int32), 2)), axis = 2)
		else:
			diffs = tf.reduce_sum(tf.reshape(diffs, (tf.shape(diffs)[0], tf.cast(tf.shape(diffs)[1] / 2, tf.int32), 2)), axis = 2)
		diffs = tf.sqrt(diffs)
		outside_points = diffs > self.point_radius
		outside_diffs = diffs[outside_points]
		self.point_outside.assign_add(tf.reduce_sum(tf.cast(outside_points, tf.float32)))
		self.all_points.assign_add(tf.cast(tf.size(diffs), tf.float32))
		self.total_outside_distance.assign_add(tf.reduce_sum(outside_diffs))

	def result(self):
		if self.accuracy:
			return 1 - self.point_outside / self.all_points

		if self.point_outside == 0:
			return 0.0
		return self.total_outside_distance / self.point_outside

	def reset_state(self):
		self.point_outside.assign(0)
		self.all_points.assign(0)
		self.total_outside_distance.assign(0)
