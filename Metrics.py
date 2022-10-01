import tensorflow as tf

def Dice(y_true_f, y_pred_f):
    y_pred_f = tf.math.argmax(y_pred_f, axis=-1)
    y_pred_f = tf.one_hot(y_pred_f, depth=2)
    smooth = 1e-20  # Используется для предотвращения знаменателя от 0.
    intersection = tf.math.reduce_sum(tf.math.reduce_sum(tf.math.multiply(y_true_f, y_pred_f), axis=1), axis=1)
    sum_sets = tf.math.reduce_sum(tf.math.reduce_sum((y_true_f), axis=1), axis=1) + \
               tf.math.reduce_sum(tf.math.reduce_sum((y_pred_f), axis=1), axis=1)
    return tf.math.reduce_mean((2. * intersection + smooth) / (sum_sets + smooth), axis=-1)