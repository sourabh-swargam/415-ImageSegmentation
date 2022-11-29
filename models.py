import tensorflow as tf
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
from segmentation_models import Unet
from keras.models import load_model
tf.keras.backend.set_image_data_format('channels_last')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def get_unet():

    tf.keras.backend.clear_session() 
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model = Unet(backbone_name='resnet34',activation='sigmoid',classes=21,input_shape=(512,512,3),encoder_weights='imagenet')


    model.load_weights('UNET_10(BEST).h5')
    return model


class convolutional_block(tf.keras.layers.Layer):
    def __init__(self, kernel=3,  filters=[4,4,8], stride=2, name="conv_block",trainable=True):
        super().__init__(name=name)
        self.F1, self.F2, self.F3 = filters
        self.kernel = kernel
        self.stride = stride
        self.conv1=Conv2D(filters=self.F1,strides=1,padding='same',kernel_size=1,kernel_initializer='he_uniform',activation='relu')
        self.bn1=BatchNormalization()
        self.act1=Activation('relu')
        self.conv2=Conv2D(filters=self.F2,strides=self.stride,padding='same',kernel_size=3,kernel_initializer='he_uniform',activation='relu')
        self.bn2=BatchNormalization()
        self.act2=Activation('relu')
        self.conv3=Conv2D(filters=self.F3,strides=1,padding='same',kernel_size=1,kernel_initializer='he_uniform',activation='relu')
        self.bn3=BatchNormalization()
        self.act3=Activation('relu')
        self.conv_parallel=Conv2D(filters=self.F3,strides=self.stride,padding='same',kernel_size=3,kernel_initializer='he_uniform',activation='relu')
        self.bn_parallel=BatchNormalization()
        self.act_parallel=Activation('relu')

        self.add=tf.keras.layers.Add()
        self.act_out=Activation('relu')

    def call(self, X):
        conv1=self.conv1(X)
        bn1=self.bn1(conv1)
        act1=self.act1(bn1)

        conv2=self.conv2(act1)
        bn2=self.bn2(conv2)
        act2=self.act2(bn2)

        conv3=self.conv3(act2)
        bn3=self.bn3(conv3)
        act3=self.act3(bn3)

        conv_parallel=self.conv_parallel(X)
        bn_parallel=self.bn_parallel(conv_parallel)
        act_parallel=self.act_parallel(bn_parallel)

        add=self.add([act3,act_parallel])
        act_out=self.act_out(add)
        X=act_out
        return X
    def get_config(self):
        cfg = super().get_config()
        return cfg 



class identity_block(tf.keras.layers.Layer):
    def __init__(self, kernel=3,  filters=[4,4,8], name="identity block",trainable=True):
        super().__init__(name=name)
        self.F1, self.F2, self.F3 = filters
        self.kernel = kernel
        self.conv1=Conv2D(filters=self.F1,strides=1,padding='same',kernel_size=1,kernel_initializer='he_uniform',activation='relu')
        self.bn1=BatchNormalization()
        self.act1=Activation('relu')
        self.conv2=Conv2D(filters=self.F2,strides=1,padding='same',kernel_size=3,kernel_initializer='he_uniform',activation='relu')
        self.bn2=BatchNormalization()
        self.act2=Activation('relu')
        self.conv3=Conv2D(filters=self.F3,strides=1,padding='same',kernel_size=1,kernel_initializer='he_uniform',activation='relu')
        self.bn3=BatchNormalization()
        self.act3=Activation('relu')

        self.add=tf.keras.layers.Add()
        self.act_out=Activation('relu')
    def call(self, X):

        conv1=self.conv1(X)
        bn1=self.bn1(conv1)
        act1=self.act1(bn1)

        conv2=self.conv2(act1)
        bn2=self.bn2(conv2)
        act2=self.act2(bn2)

        conv3=self.conv3(act2)
        bn3=self.bn3(conv3)
        act3=self.act3(bn3)

        add=self.add([act3,X])
        out=self.act_out(add)
        return X
    def get_config(self):
        cfg = super().get_config()
        return cfg

def create_identity_blocks(conv_output,count,stride,filters,name):
  temp=conv_output
  identity_block_=None
  for i in range(count):
    identity_name=name+'_'+str(i+1)
    identity_block_=identity_block(kernel=3,filters=filters, name=identity_name)
    identity_block_out=identity_block_(temp)
    temp=identity_block_out
  identity_block_ouput=temp
  return identity_block_ouput 


class global_flow(tf.keras.layers.Layer):
    def __init__(self, name="global_flow",trainable=True):
        super().__init__(name=name)
        self.global_avg=GlobalAveragePooling2D(data_format='channels_last')
        self.reshape=tf.keras.layers.Reshape((1,1,64))
        self.bn=BatchNormalization()
        self.act=Activation('relu')
        self.conv=Conv2D(64,kernel_size=1,strides=1,padding='same',kernel_initializer='he_uniform',activation='relu')
        self.upsample=UpSampling2D((64,64),interpolation='bilinear')
    def call(self, X):
        global_avg=self.global_avg(X)
        reshape=self.reshape(global_avg)
        bn=self.bn(reshape)
        act=self.act(bn)
        conv=self.conv(act)
        upsample=self.upsample(conv)
        X=upsample
        return X
    def get_config(self):
        cfg = super().get_config()
        return cfg 

class context_flow(tf.keras.layers.Layer):    
    def __init__(self, name="context_flow",trainable=True):
        super().__init__(name=name)
        self.concat=tf.keras.layers.Concatenate(axis=-1)
        self.avg_pool=AveragePooling2D(2)
        self.conv1=Conv2D(32,kernel_size=3,padding='same',activation='relu',kernel_initializer='he_uniform')
        self.conv2=Conv2D(64,kernel_size=3,padding='same',activation='relu',kernel_initializer='he_uniform')
        self.conv3=Conv2D(32,kernel_size=1,padding='same',activation='relu',kernel_initializer='he_uniform')
        self.conv4=Conv2D(64,kernel_size=1,padding='same',activation='relu',kernel_initializer='he_uniform')
        self.act1=Activation('relu')
        self.act2=Activation('sigmoid')
        self.multiply=tf.keras.layers.Multiply()
        self.add=tf.keras.layers.Add()
        self.upsample=UpSampling2D(2)
    def call(self, X):
        INP, FLOW = X[0], X[1] 
        concat=self.concat([INP,FLOW])
        avg_pool=self.avg_pool(concat)
        conv1=self.conv1(avg_pool)
        conv2=self.conv2(conv1)

        conv3=self.conv3(conv2)
        act1=self.act1(conv3)
        conv4=self.conv4(act1)
        act2=self.act2(conv4)

        multiply=self.multiply([conv2,act2])
        add=self.add([conv2,multiply])

        X=self.upsample(add)

        return X
    def get_config(self):
        cfg = super().get_config()
        return cfg 


class sum_fsm(tf.keras.layers.Layer):    
    def __init__(self, name="sum_fsm",trainable=True):
        super().__init__(name=name)
        self.add=tf.keras.layers.Add()
    def call(self, X):
        X=self.add([X[0],X[1],X[2],X[3]])
        return X
    def get_config(self):
        cfg = super().get_config()
        return cfg 

class fsm(tf.keras.layers.Layer):    
    def __init__(self, name="feature_selection",trainable=True):
        super().__init__(name=name)
        self.conv=Conv2D(64,kernel_size=3,padding='same',activation='relu',kernel_initializer='he_uniform')
        self.global_max=GlobalMaxPooling2D()
        self.reshape=tf.keras.layers.Reshape((1,1,64))
        self.bn=BatchNormalization()
        self.act=Activation('sigmoid')

        self.multiply=tf.keras.layers.Multiply()
        self.upsample=UpSampling2D(2)
    def call(self, X):
        conv=self.conv(X)
        global_max=self.global_max(conv)
        reshape=self.reshape(global_max)
        bn=self.bn(reshape)
        act=self.act(bn)

        multiply=self.multiply([X,act])

        FSM_Conv_T=self.upsample(multiply)
        return FSM_Conv_T
    def get_config(self):
        cfg = super().get_config()
        return cfg 



class agcn(tf.keras.layers.Layer):    
    def __init__(self, name="global_conv_net",trainable=True):
        super().__init__(name=name)
        self.conv1=Conv2D(32,kernel_size=(7,1),padding='same',activation='relu',kernel_initializer='he_uniform')
        self.conv2=Conv2D(64,kernel_size=(1,7),padding='same',activation='relu',kernel_initializer='he_uniform')
        self.conv3=Conv2D(32,kernel_size=(1,7),padding='same',activation='relu',kernel_initializer='he_uniform')
        self.conv4=Conv2D(64,kernel_size=(7,1),padding='same',activation='relu',kernel_initializer='he_uniform')
        self.conv5=Conv2D(64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_uniform')

        self.add1=tf.keras.layers.Add()
        self.add2=tf.keras.layers.Add()
    def call(self, X):
        # please implement the above mentioned architecture
        conv1=self.conv1(X)
        conv2=self.conv2(conv1)

        conv3=self.conv3(X)
        conv4=self.conv4(conv3)

        add1=self.add1([conv2,conv4])
        conv5=self.conv5(add1)

        X=self.add2([add1,conv5])
        return X
    def get_config(self):
        cfg = super().get_config()
        return cfg 


def get_canet():
    X_input = Input(shape=(512,512,3),name='Input')

    # Stage 1
    X = Conv2D(64, (3, 3), name='conv1', padding="same",strides=2, kernel_initializer='he_uniform',activation='relu')(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu', name='activation_conv1')(X)

    conv_block1=convolutional_block(kernel=3,filters=[4,4,8], stride=2, name="conv_block1")
    conv_block1_out=conv_block1(X)
    identity_block1_ouput=create_identity_blocks(conv_block1_out,count=1,stride=2,filters=[4,4,8],name='identity_block1')


    conv_block2=convolutional_block(kernel=3,filters=[8,8,16], stride=2, name="conv_block2")
    conv_block2_out=conv_block2(identity_block1_ouput)
    identity_block2_ouput=create_identity_blocks(conv_block2_out,count=2,stride=2,filters=[8,8,16],name='identity_block2')


    conv_block3=convolutional_block(kernel=3,filters=[16,16,32], stride=1, name="conv_block3")
    conv_block3_out=conv_block3(identity_block2_ouput)
    identity_block3_ouput=create_identity_blocks(conv_block3_out,count=3,stride=1,filters=[16,16,32],name='identity_block3')


    conv_block4=convolutional_block(kernel=3,filters=[32,32,64], stride=1, name="conv_block4")
    conv_block4_out=conv_block4(identity_block3_ouput)
    identity_block4_ouput=create_identity_blocks(conv_block4_out,count=4,stride=1,filters=[32,32,64],name='identity_block4')

    GF=global_flow()
    GF_output=GF(identity_block4_ouput)
    context_flow1=context_flow(name='context_flow1')
    context_flow1_output=context_flow1([identity_block4_ouput,GF_output])

    context_flow2=context_flow(name='context_flow2')
    context_flow2_output=context_flow2([identity_block4_ouput,context_flow1_output])

    context_flow3=context_flow(name='context_flow3')
    context_flow3_output=context_flow3([identity_block4_ouput,context_flow2_output])


    sum_fsm_block=sum_fsm()
    fsm_input=sum_fsm_block([GF_output,context_flow1_output,context_flow2_output,context_flow3_output])

    fsm_block=fsm()
    fsm_output=fsm_block(fsm_input)
    agcn_block=agcn()
    agcn_output=agcn_block(identity_block1_ouput)

    X=tf.keras.layers.Concatenate(axis=-1, name='concat_agcn_fsm')([fsm_output,agcn_output])
    #X=Conv2D(64,kernel_size=(7,7),padding='same',activation='relu',kernel_initializer='he_uniform', name='conv2_1')(X)
    X=Conv2D(32,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_uniform', name='conv2_1')(X)
    X=Conv2D(21,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_uniform', name='conv2_2')(X)
    X=UpSampling2D(4, name='upsample_output')(X)
    output=Activation('softmax', name='output')(X)
    tf.keras.backend.clear_session() 
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model=Model(X_input,output)

    model.load_weights('CNET_15 - Copy.h5')

    return model



