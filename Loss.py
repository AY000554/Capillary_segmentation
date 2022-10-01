import tensorflow as tf
from tensorflow import keras

def Dice_loss(y_true_f, y_pred_f):
    smooth = 1e-25  # Используется для предотвращения знаменателя от 0.
    intersection = tf.math.reduce_sum(tf.math.reduce_sum(tf.math.multiply(y_true_f, y_pred_f), axis=1), axis=1)
    sum_sets = tf.math.reduce_sum(tf.math.reduce_sum((y_true_f), axis=1), axis=1) + \
               tf.math.reduce_sum(tf.math.reduce_sum((y_pred_f), axis=1), axis=1)
    return 1. - tf.math.reduce_mean((2. * intersection + smooth) / (sum_sets + smooth))

cce = tf.keras.losses.CategoricalCrossentropy()
def Mix_loss_dice_and_CCE(y_true_f, y_pred_f):
  k = 0.3
  return (k * Dice_loss(y_true_f, y_pred_f)) + ((1 - k) * cce(y_true_f, y_pred_f))