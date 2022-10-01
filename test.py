from tensorflow.keras.optimizers import Adam
from tensorflow import keras as K
from datetime import datetime
import numpy as np
import os
import cv2
import time

from Metrics import *
from build_model import build_model
from Loss import Mix_loss_dice_and_CCE
from Data_generator import DataGenerator, split_data

os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if len(gpus) > 0:
    try:
        for i in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)
    except RuntimeError: pass

if __name__ == "__main__":
    test_img_folder = r'/Data/test_filtred_dataset_mc/imgs'
    result_mask_folder = r'/Data/test_filtred_dataset_mc/masks'
    path_model = r"ep0434-U-Res-net_7_512__64_16__mixloss_0_3.h5"
    opt = Adam(learning_rate=1e-4)
    categorical_crossentropy_loss = tf.keras.losses.CategoricalCrossentropy()
    model = K.models.load_model(path_model,
                                custom_objects={'optimizer': opt,
                                                'Mix_loss_dice_and_CCE': Mix_loss_dice_and_CCE,
                                                'Dice': Dice,
                                                },
                                compile=True)

    model.summary()
    test_list = os.listdir(test_img_folder)
    if not(os.path.exists(result_mask_folder)):
        os.makedirs(result_mask_folder)
    time_list = []
    count_experiments = 23
    for path_img in test_list:
        img = cv2.imread(os.path.join(test_img_folder, path_img))
        img = np.divide(img[:, 0:1616], 255, dtype=np.float32)
        t1 = time.time()

        height, width = img.shape[:2]
        center = (width / 2, height / 2)
        rand_angle = np.array([-45, -42, -40, -37, -35, -32, -30, -27, -25, -22, -20, 0, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45])
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rand_angle[0], scale=1)
        rotated_image = cv2.warpAffine(
            src=img, M=rotate_matrix, dsize=(width, height))
        pred_mask_rot = model.predict(np.expand_dims(rotated_image, axis=0))
        re_rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rand_angle[0] * -1, scale=1)
        pred_mask = cv2.warpAffine(
            src=pred_mask_rot[0, :, :, :], M=re_rotate_matrix, dsize=(width, height))
        for i in range(count_experiments-1):
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rand_angle[i+1], scale=1)
            rotated_image = cv2.warpAffine(
                src=img, M=rotate_matrix, dsize=(width, height))

            pred_mask_rot = model.predict(np.expand_dims(rotated_image, axis=0))

            re_rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rand_angle[i+1] * -1, scale=1)
            pred_mask += cv2.warpAffine(
                src=pred_mask_rot[0, :, :, :], M=re_rotate_matrix, dsize=(width, height))
        time_list.append(time.time() - t1)
        pred_mask = np.argmax(pred_mask, axis=-1)
        pred_mask = np.uint8(pred_mask) * 255
        pred_mask = pred_mask[:, :]
        add_block = np.repeat(np.expand_dims(pred_mask[:, -1], axis=1), 8, axis=-1)
        result_mask = np.concatenate([pred_mask, add_block], axis=1)
        cv2.imwrite(os.path.join(result_mask_folder, path_img), np.expand_dims(result_mask, -1),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

    print("Mean time processing one image: ", np.array(time_list).mean())
    print('Complated')
