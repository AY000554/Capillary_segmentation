import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

def split_data(path_data="", mask_folder =""):
    path_masks = os.path.join(path_data, mask_folder)
    masks_name_list = os.listdir(path_masks)
    # Разбиение на обуч. и пров. выборки
    train_data, val_data, train_files, val_files = train_test_split(masks_name_list, masks_name_list, test_size=0.15,
                                                                    shuffle=False, random_state=1)
    return train_data, val_data


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_name_files,
                 path_data,
                 img_folder="train_filtred_dataset_mc\imgs",
                 mask_folder="train_filtred_dataset_mc\masks",
                 batch_size=4,
                 im_size=(512, 512),
                 shuffle=True,
                 augmentation=False
                 ):
        self.batch_size = batch_size
        self.data_list = list_name_files
        self.path_img = os.path.join(path_data, img_folder)
        self.path_masks = os.path.join(path_data, mask_folder)
        self.shuffle = shuffle
        self.im_size = im_size
        self.on_epoch_end()
        self.augmentation = augmentation
        if self.augmentation:
            self.aug = iaa.Affine(rotate=(-45, 45)) 

    def __len__(self):
        return int(np.floor(len(self.data_list) / self.batch_size))

    def random_crop(self, img, mask):
        height, width = img.shape[0], mask.shape[1]
        dy, dx = self.im_size[0], self.im_size[1]
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y + dy), x:(x + dx), :], mask[y:(y + dy), x:(x + dx), :]

    def __getitem__(self, index):
        start_ind = index * self.batch_size
        end_ind = (index + 1) * self.batch_size
        if end_ind >= len(self.indexes):
            indexes = self.indexes[start_ind:]
        else:
            indexes = self.indexes[start_ind: end_ind]

        imgs = np.zeros((len(indexes), self.im_size[0], self.im_size[1], 3), dtype=np.float32)
        masks = np.zeros((len(indexes), self.im_size[0], self.im_size[1], 2), dtype=np.float32)

        for sample_ind, ind in enumerate(indexes):
            imgs[sample_ind, ...], masks[sample_ind, ...] = self.__getsample__(ind)
        return imgs, masks
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getsample__(self, idx):
        try:
            img = cv2.imread(os.path.join(self.path_img, self.data_list[idx]))
        except:
            print("-" * 100)
            print("Не найдено изображение по указанному пути: " + '"' + os.path.join(self.path_img,
                                                                                     self.data_list[idx]) + '"')
            print("-" * 100)
        # Нормализация
        img = np.divide(img, 255, dtype=np.float32)
        try:
            mask = cv2.imread(os.path.join(self.path_masks, self.data_list[idx]),  cv2.IMREAD_GRAYSCALE)
        except:
            print("-" * 100)
            print("Не найдено изображение по указанному пути: " + '"' + os.path.join(self.path_masks,
                                                                                     self.data_list[idx]) + '"')
            print("-" * 100)
        mask = np.expand_dims(mask, -1)
        img_crop, mask_crop = self.random_crop(img, mask)
        if self.augmentation:
            segmap = SegmentationMapsOnImage(np.int32(mask_crop), shape=mask_crop.shape)
            im_aug, segmap_out = self.aug(image=img_crop, segmentation_maps=segmap)
            img_crop = im_aug
            mask_crop = segmap_out.arr

        mask_crop = mask_crop.clip(0, 1)
        mask_crop = tf.keras.utils.to_categorical(mask_crop, num_classes=2)
        return img_crop, mask_crop