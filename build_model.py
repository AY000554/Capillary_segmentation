import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate, add, ReLU, BatchNormalization, MaxPool2D, Activation, SpatialDropout2D

def build_model(feature_size):
    drop_koef = 0.3
    ecoding_chanel = feature_size
    decoding_chanel = int(feature_size / 4)
    params = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': tf.keras.initializers.glorot_uniform()}
    params_not_activation = { 'padding': 'same', 'kernel_initializer': tf.keras.initializers.glorot_uniform()}
    
    input = Input(shape=(None, None, 3), name='img')
    
    conv1_1 = Conv2D(ecoding_chanel, 3, **params, name='conv1_1')(input)
    
    conv1_2 = Conv2D(ecoding_chanel, 3, **params_not_activation, name='conv1_2')(conv1_1)
    conv1_2 = ReLU()(BatchNormalization()(conv1_2))
    conv1_3 = Conv2D(ecoding_chanel, 3, **params_not_activation, name='conv1_3')(conv1_2)
    res1_1 = ReLU()(add([conv1_1, BatchNormalization()(conv1_3)]))
    
    conv1_4 = Conv2D(ecoding_chanel, 3, **params_not_activation, name='conv1_4')(res1_1)
    conv1_4 = ReLU()(BatchNormalization()(conv1_4))
    conv1_5 = Conv2D(ecoding_chanel, 3, **params_not_activation, name='conv1_5')(conv1_4)
    res1_2 = ReLU()(add([res1_1, BatchNormalization()(conv1_5)]))
    
    conv1_6 = Conv2D(ecoding_chanel, 3, **params_not_activation, name='conv1_6')(res1_2)
    conv1_6 = ReLU()(BatchNormalization()(conv1_6))
    conv1_7 = Conv2D(ecoding_chanel, 3, **params_not_activation, name='conv1_7')(conv1_6)
    res1_3 = ReLU()(add([res1_2, BatchNormalization()(conv1_7)]))
    
    
    pool_1  = MaxPool2D(pool_size=(2, 2), name='pool_1', strides=(2, 2))(res1_3)
    
    conv2_1 = Conv2D(ecoding_chanel*2, 3, **params_not_activation, name='conv2_1')(pool_1)
    conv2_1 = ReLU()(BatchNormalization()(conv2_1))
    
    conv2_2 = Conv2D(ecoding_chanel*2, 3, **params_not_activation, name='conv2_2')(conv2_1)
    conv2_2 = ReLU()(BatchNormalization()(conv2_2))
    conv2_3 = Conv2D(ecoding_chanel*2, 3, **params_not_activation, name='conv2_3')(conv2_2)
    res2_1 = ReLU()(add([conv2_1, BatchNormalization()(conv2_3)]))
    
    conv2_4 = Conv2D(ecoding_chanel*2, 3, **params_not_activation, name='conv2_4')(res2_1)
    conv2_4 = ReLU()(BatchNormalization()(conv2_4))
    conv2_5 = Conv2D(ecoding_chanel*2, 3, **params_not_activation, name='conv2_5')(conv2_4)
    res2_2 = ReLU()(add([res2_1, BatchNormalization()(conv2_5)]))
    
    conv2_6 = Conv2D(ecoding_chanel*2, 3, **params_not_activation, name='conv2_6')(res2_2)
    conv2_6 = ReLU()(BatchNormalization()(conv2_6))
    conv2_7 = Conv2D(ecoding_chanel*2, 3, **params_not_activation, name='conv2_7')(conv2_6)
    res2_3 = ReLU()(add([res2_2, BatchNormalization()(conv2_7)]))
    
    
    pool_2  = MaxPool2D(pool_size=(2, 2), name='pool_2', strides=(2, 2))(res2_3)
    
    conv3_1 = Conv2D(ecoding_chanel*4, 3, **params_not_activation, name='conv3_1')(pool_2)
    conv3_1 = ReLU()(BatchNormalization()(conv3_1))
    
    conv3_2 = Conv2D(ecoding_chanel*4, 3, **params_not_activation, name='conv3_2')(conv3_1)
    conv3_2 = ReLU()(BatchNormalization()(conv3_2))
    conv3_3 = Conv2D(ecoding_chanel*4, 3, **params_not_activation, name='conv3_3')(conv3_2)
    res3_1 = ReLU()(add([conv3_1, BatchNormalization()(conv3_3)]))
    
    conv3_4 = Conv2D(ecoding_chanel*4, 3, **params_not_activation, name='conv3_4')(res3_1)
    conv3_4 = ReLU()(BatchNormalization()(conv3_4))
    conv3_5 = Conv2D(ecoding_chanel*4, 3, **params_not_activation, name='conv3_5')(conv3_4)
    res3_2 = ReLU()(add([res3_1, BatchNormalization()(conv3_5)]))
    
    conv3_6 = Conv2D(ecoding_chanel*4, 3, **params_not_activation, name='conv3_6')(res3_2)
    conv3_6 = ReLU()(BatchNormalization()(conv3_6))
    conv3_7 = Conv2D(ecoding_chanel*4, 3, **params_not_activation, name='conv3_7')(conv3_6)
    res3_3 = ReLU()(add([res3_2, BatchNormalization()(conv3_7)]))
    
    
    pool_3  = MaxPool2D(pool_size=(2, 2), name='pool_3', strides=(2, 2))(res3_3)
    
    conv4_1 = Conv2D(ecoding_chanel*8, 3, **params_not_activation, name='conv4_1')(pool_3)
    conv4_1 = ReLU()(BatchNormalization()(conv4_1))
    
    conv4_2 = Conv2D(ecoding_chanel*8, 3, **params_not_activation, name='conv4_2')(conv4_1)
    conv4_2 = ReLU()(BatchNormalization()(conv4_2))
    conv4_3 = Conv2D(ecoding_chanel*8, 3, **params_not_activation, name='conv4_3')(conv4_2)
    res4_1 = ReLU()(add([conv4_1, BatchNormalization()(conv4_3)]))
    
    conv4_4 = Conv2D(ecoding_chanel*8, 3, **params_not_activation, name='conv4_4')(res4_1)
    conv4_4 = ReLU()(BatchNormalization()(conv4_4))
    conv4_5 = Conv2D(ecoding_chanel*8, 3, **params_not_activation, name='conv4_5')(conv4_4)
    res4_2 = ReLU()(add([res4_1, BatchNormalization()(conv4_5)]))
    
    conv4_6 = Conv2D(ecoding_chanel*8, 3, **params_not_activation, name='conv4_6')(res4_2)
    conv4_6 = ReLU()(BatchNormalization()(conv4_6))
    conv4_7 = Conv2D(ecoding_chanel*8, 3, **params_not_activation, name='conv4_7')(conv4_6)
    res4_3 = ReLU()(add([res4_2, BatchNormalization()(conv4_7)]))
    
    
    pool_4  = MaxPool2D(pool_size=(2, 2), name='pool_4', strides=(2, 2))(res4_3)
    
    conv5_1 = Conv2D(ecoding_chanel*16, 3, **params_not_activation, name='conv5_1')(pool_4)
    conv5_1 = ReLU()(BatchNormalization()(conv5_1))
    
    conv5_2 = Conv2D(ecoding_chanel*16, 3, **params_not_activation, name='conv5_2')(conv5_1)
    conv5_2 = ReLU()(BatchNormalization()(conv5_2))
    conv5_3 = Conv2D(ecoding_chanel*16, 3, **params_not_activation, name='conv5_3')(conv5_2)
    res5_1 = ReLU()(add([conv5_1, BatchNormalization()(conv5_3)]))
    
    conv5_4 = Conv2D(ecoding_chanel*16, 3, **params_not_activation, name='conv5_4')(res5_1)
    conv5_4 = ReLU()(BatchNormalization()(conv5_4))
    conv5_5 = Conv2D(ecoding_chanel*16, 3, **params_not_activation, name='conv5_5')(conv5_4)
    res5_2 = ReLU()(add([res5_1, BatchNormalization()(conv5_5)]))
    
    conv5_6 = Conv2D(ecoding_chanel*16, 3, **params_not_activation, name='conv5_6')(res5_2)
    conv5_6 = ReLU()(BatchNormalization()(conv5_6))
    conv5_7 = Conv2D(ecoding_chanel*16, 3, **params_not_activation, name='conv5_7')(conv5_6)
    res5_3 = ReLU()(add([res5_2, BatchNormalization()(conv5_7)]))
    
    
    up4 = Conv2D(decoding_chanel*8, 2, **params, name='conv__up4')(UpSampling2D(size = (2,2), name='up4__res5_3')(res5_3))
    merge4 = concatenate([res4_3,up4], axis = 3, name='merge4__res4_3_up4')
    deconv4_7 = Conv2D(decoding_chanel*8, 3, **params_not_activation, name='deconv4_7')(SpatialDropout2D(drop_koef)(merge4))
    deconv4_7 = Activation('relu')(BatchNormalization()(deconv4_7))
    
    deconv4_6 = Conv2D(decoding_chanel*8, 3, **params_not_activation, name='deconv4_6')(deconv4_7)
    deconv4_6 = Activation('relu')(BatchNormalization()(deconv4_6))
    deconv4_5 = Conv2D(decoding_chanel*8, 3, **params_not_activation, name='deconv4_5')(deconv4_6)
    de_res4_3 = ReLU(name='de_res4_3')(add([deconv4_7, BatchNormalization()(deconv4_5)]))
    
    deconv4_4 = Conv2D(decoding_chanel*8, 3, **params_not_activation, name='deconv4_4')(de_res4_3)
    deconv4_4 = Activation('relu')(BatchNormalization()(deconv4_4))
    deconv4_3 = Conv2D(decoding_chanel*8, 3, **params_not_activation, name='deconv4_3')(deconv4_4)
    de_res4_2 = ReLU(name='de_res4_2')(add([de_res4_3, BatchNormalization()(deconv4_3)]))
    
    deconv4_2 = Conv2D(decoding_chanel*8, 3, **params_not_activation, name='deconv4_2')(de_res4_2)
    deconv4_2 = Activation('relu')(BatchNormalization()(deconv4_2))
    deconv4_1 = Conv2D(decoding_chanel*8, 3, **params_not_activation, name='deconv4_1')(deconv4_2)
    de_res4_1 = ReLU(name='de_res4_1')(add([de_res4_2, BatchNormalization()(deconv4_1)]))
    
    
    up3 = Conv2D(decoding_chanel*4, 2, **params, name='conv__up3')(UpSampling2D(size = (2,2), name='up3__res4_1')(de_res4_1))
    merge3 = concatenate([res3_3,up3], axis = 3, name='merge3__res3_3_up3')
    deconv3_7 = Conv2D(decoding_chanel*4, 3, **params_not_activation, name='deconv3_7')(SpatialDropout2D(drop_koef)(merge3))
    deconv3_7 = Activation('relu')(BatchNormalization()(deconv3_7))
    
    deconv3_6 = Conv2D(decoding_chanel*4, 3, **params_not_activation, name='deconv3_6')(deconv3_7)
    deconv3_6 = Activation('relu')(BatchNormalization()(deconv3_6))
    deconv3_5 = Conv2D(decoding_chanel*4, 3, **params_not_activation, name='deconv3_5')(deconv3_6)
    de_res3_3 = ReLU(name='de_res3_3')(add([deconv3_7, BatchNormalization()(deconv3_5)]))
    
    deconv3_4 = Conv2D(decoding_chanel*4, 3, **params_not_activation, name='deconv3_4')(de_res3_3)
    deconv3_4 = Activation('relu')(BatchNormalization()(deconv3_4))
    deconv3_3 = Conv2D(decoding_chanel*4, 3, **params_not_activation, name='deconv3_3')(deconv3_4)
    de_res3_2 = ReLU(name='de_res3_2')(add([de_res3_3, BatchNormalization()(deconv3_3)]))
    
    deconv3_2 = Conv2D(decoding_chanel*4, 3, **params_not_activation, name='deconv3_2')(de_res3_2)
    deconv3_2 = Activation('relu')(BatchNormalization()(deconv3_2))
    deconv3_1 = Conv2D(decoding_chanel*4, 3, **params_not_activation, name='deconv3_1')(deconv3_2)
    de_res3_1 = ReLU(name='de_res3_1')(add([de_res3_2, BatchNormalization()(deconv3_1)]))
    
    
    up2 = Conv2D(decoding_chanel*2, 2, **params, name='conv__up2')(UpSampling2D(size = (2,2), name='up2__de_res3_1')(de_res3_1))
    merge2 = concatenate([res2_3,up2], axis = 3, name='merge2__res2_3_up2')
    deconv2_7 = Conv2D(decoding_chanel*2, 3, **params_not_activation, name='deconv2_7')(SpatialDropout2D(drop_koef)(merge2))
    deconv2_7 = Activation('relu')(BatchNormalization()(deconv2_7))
    
    deconv2_6 = Conv2D(decoding_chanel*2, 3, **params_not_activation, name='deconv2_6')(deconv2_7)
    deconv2_6 = Activation('relu')(BatchNormalization()(deconv2_6))
    deconv2_5 = Conv2D(decoding_chanel*2, 3, **params_not_activation, name='deconv2_5')(deconv2_6)
    de_res2_3 = ReLU(name='de_res2_3')(add([deconv2_7, BatchNormalization()(deconv2_5)]))
    
    deconv2_4 = Conv2D(decoding_chanel*2, 3, **params_not_activation, name='deconv2_4')(de_res2_3)
    deconv2_4 = Activation('relu')(BatchNormalization()(deconv2_4))
    deconv2_3 = Conv2D(decoding_chanel*2, 3, **params_not_activation, name='deconv2_3')(deconv2_4)
    de_res2_2 = ReLU(name='de_res2_2')(add([de_res2_3, BatchNormalization()(deconv2_3)]))
    
    deconv2_2 = Conv2D(decoding_chanel*2, 3, **params_not_activation, name='deconv2_2')(de_res2_2)
    deconv2_2 = Activation('relu')(BatchNormalization()(deconv2_2))
    deconv2_1 = Conv2D(decoding_chanel*2, 3, **params_not_activation, name='deconv2_1')(deconv2_2)
    de_res2_1 = ReLU(name='de_res2_1')(add([de_res2_2, BatchNormalization()(deconv2_1)]))
    
    
    up1 = Conv2D(decoding_chanel, 2, **params, name='conv__up1')(UpSampling2D(size = (2,2), name='up1__de_res2_1')(de_res2_1))
    merge1 = concatenate([res1_3,up1], axis = 3, name='merge1__res1_3_up1')
    deconv1_7 = Conv2D(decoding_chanel, 3, **params_not_activation, name='deconv1_7')(SpatialDropout2D(drop_koef)(merge1))
    deconv1_7 = Activation('relu')(BatchNormalization()(deconv1_7))
    
    deconv1_6 = Conv2D(decoding_chanel, 3, **params_not_activation, name='deconv1_6')(deconv1_7)
    deconv1_6 = Activation('relu')(BatchNormalization()(deconv1_6))
    deconv1_5 = Conv2D(decoding_chanel, 3, **params_not_activation, name='deconv1_5')(deconv1_6)
    de_res1_3 = ReLU(name='de_res1_3')(add([deconv1_7, BatchNormalization()(deconv1_5)]))
    
    deconv1_4 = Conv2D(decoding_chanel, 3, **params_not_activation, name='deconv1_4')(de_res1_3)
    deconv1_4 = Activation('relu')(BatchNormalization()(deconv1_4))
    deconv1_3 = Conv2D(decoding_chanel, 3, **params_not_activation, name='deconv1_3')(deconv1_4)
    de_res1_2 = ReLU(name='de_res1_2')(add([de_res1_3, BatchNormalization()(deconv1_3)]))
    
    deconv1_2 = Conv2D(decoding_chanel, 3, **params_not_activation, name='deconv1_2')(de_res1_2)
    deconv1_2 = Activation('relu')(BatchNormalization()(deconv1_2))
    deconv1_1 = Conv2D(decoding_chanel, 3, **params_not_activation, name='deconv1_1')(deconv1_2)
    de_res1_1 = ReLU(name='de_res1_1')(add([de_res1_2, BatchNormalization()(deconv1_1)]))
    
    output = Conv2D(2, 1, activation=tf.keras.activations.softmax, padding='same', name='output')(de_res1_1)
    
    model = tf.keras.Model(inputs = input, outputs = output)
    
    return model