# TensorFlow tutorials
#
# https://www.tensorflow.org/get_started/feature_columns
# This is pseudocode, not able to run.
import tensorflow as tf


def make_dataset(latitude, longitude, labels):
    assert latitude.shape == longitude.shape == labels.shape

    features = {'latitude':latitude.flatten(),
                'longitude':longitude.flatten()}
    labels = labels.flatten()

    return tf.data.Dataset.from_tensor_slices((features, labels))


# Bucketize the latitude and longitude using the 'edges'
latitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('latitude'),
    list(atlanta.latitude.edges))

longitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('longitude'),
    list(atlanta.longitude.edges))

# Cross the bucketized columns, using 5000 hash bins.
crossed_lat_lon_fc = tf.feature_column.crossed_column(
    [latitude_bucket_fc, longitude_bucket_fc], 5000)

# Final feature column
fc = [latitude_bucket_fc, longitude_bucket_fc, crossed_lat_lon_fc]

# Build and train the Estimator.
est = tf.estimator.LinearRegressor(fc, ...)

