import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import os
import numpy as np

from Data_generator import DataGenerator, split_data
from build_model import build_model
from Metrics import *
from Loss import Mix_loss_dice_and_CCE

os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if len(gpus) > 0:
    try:
        for i in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)
    except RuntimeError: pass

def train():
    path_data = r"/Data"
    img_folder = r'train_filtred_dataset_mc/imgs'
    mask_folder = r'train_filtred_dataset_mc/masks'
    batch_size = 4
    start_lr = 1e-4
    train_data_list, val_data_list = split_data(path_data, mask_folder)
    train_datagen = DataGenerator(list_name_files = train_data_list,
                                  path_data = path_data,
                                  img_folder = img_folder,
                                  mask_folder= mask_folder,
                                  batch_size=batch_size,
                                  im_size=(512, 512),
                                  shuffle=True,
                                  augmentation=True
                                )

    val_datagen = DataGenerator(list_name_files = val_data_list,
                                  path_data = path_data,
                                  img_folder = img_folder,
                                  mask_folder= mask_folder,
                                  batch_size=batch_size,
                                  im_size=(512, 512),
                                  shuffle=False,
                                  augmentation=False
                                )
    
    model = build_model(feature_size=64)
    opt = Adam(learning_rate=start_lr)
    METRICS = [Dice]
    model.compile(optimizer = opt, loss=Mix_loss_dice_and_CCE, metrics = METRICS)
    model.summary(positions=[.33, .60, .74, 1.])
    logdir = os.path.join("logs", datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
    check_point_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(check_point_dir)
    
    check_point_cbk = K.callbacks.ModelCheckpoint(
        os.path.join(check_point_dir, "ep{epoch:04d}-val_Dice{val_Dice:.7f}.h5"),
        monitor='val_Dice',
        verbose=0, 
        save_best_only=True,
        mode='max', 
        save_freq='epoch')
    
    tensorboard_cbk = K.callbacks.TensorBoard(
            log_dir = logdir,                                  
            histogram_freq = 0,                                   
            write_grads = False,                              
            update_freq = 'epoch',                                   
            write_graph = True) 

    end_learning_rate = 0
    decay_steps = 500
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        start_lr,
        decay_steps,
        end_learning_rate,
        power=0.7)
    change_lr_cbk = tf.keras.callbacks.LearningRateScheduler(learning_rate_fn)
    
    csv_log_cbk = K.callbacks.CSVLogger(
            filename = os.path.join(logdir, "log_train.csv"),                                                           
            separator = ';',                                  
            append = True)
    
    model.fit(train_datagen,
              validation_data = val_datagen,
              use_multiprocessing = True,
              max_queue_size = 128,
              workers = 8,
              epochs = 500,
              callbacks = [tensorboard_cbk, change_lr_cbk, csv_log_cbk, check_point_cbk], 
              shuffle = True)                      
                                
if __name__ == '__main__':
    train()
    
    
    
