import tensorflow as tf
from tensorflow.keras.layers import Conv2D as Conv
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
'''
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
x_train=np.array(x_train)/255
x_test=np.array(x_test)/255
patches=tf.image.extract_patches(x_train[:1],sizes=[1,8,8,1],strides=[1,8,8,1],rates=[1,1,1,1],padding="VALID")
'''
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,mlp_dim,embed_dim,num_heads,dropout=0.1):
        super(TransformerBlock,self).__init__()
        self.Multihead=MultiHeadAttention(embed_dim,num_heads)
        self.LayerNorm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.LayerNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.Dropout1=tf.keras.layers.Dropout(dropout)
        self.Dropout2 = tf.keras.layers.Dropout(dropout)
        self.mlp=tf.keras.Sequential([Dense(mlp_dim,activation="relu"),Dense(embed_dim)])
    def call(self,inp,training):
        x=self.LayerNorm1(inp)
        x=self.Multihead(x)
        x=self.Dropout1(x)
        a=self.LayerNorm1(x+inp)
        inp1=self.mlp(a)
        inp1=self.Dropout2(inp1,training=training)
        v=self.LayerNorm2(inp1+a)
        return v
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,embed_dim,num_heads):
        super(MultiHeadAttention,self).__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.query_dense=Dense(embed_dim)
        self.key_dense=Dense(embed_dim)
        self.value_dense=Dense(embed_dim)
        self.combine_dense=Dense(embed_dim)
        self.proj_dim=embed_dim//num_heads
    def attention(self,query,key,value):
        score=tf.matmul(query,key,transpose_b=True)
        dim=tf.cast(tf.shape(key)[-1],tf.float32)
        score=score/tf.sqrt(dim)
        activation=tf.nn.softmax(score,axis=-1)
        total=tf.matmul(activation,value)
        return total,activation
    def head_reshape(self,x,batch_size):
        x=tf.reshape(x,(batch_size,-1,self.num_heads,self.proj_dim))
        x=tf.transpose(x,perm=[0,2,1,3])
        return x
    def call(self,data):
        batch_size=tf.shape(data)[0]
        query=self.query_dense(data)
        key=self.key_dense(data)
        value=self.value_dense(data)
        query=self.head_reshape(query,batch_size)
        key = self.head_reshape(key, batch_size)
        value = self.head_reshape(value, batch_size)
        attention,weights=self.attention(query,key,value)
        attention=tf.transpose(attention,perm=[0,2,1,3])
        concat_attention=tf.reshape(attention,[batch_size,-1,self.embed_dim])
        attention=self.combine_dense(concat_attention)
        return attention

class VisionTransformer(tf.keras.Model):
    def __init__(self,class_number,num_layers,num_heads,image_size,mlp_dim,patch_size,d_model,channels=3,dropout=0.1):
        super(VisionTransformer,self).__init__()
        self.class_number=class_number
        self.num_layers=num_layers
        self.d_model=d_model
        self.image_size=image_size
        self.patch_size=patch_size
        self.num_heads=num_heads
        self.channels=channels
        self.rescale=Rescaling(1./255)
        num_patches = self.create_patche(image_size, patch_size, channels)
        self.patch_proj=self.create_pos_encoding(d_model,num_patches=num_patches)
        self.dropout=dropout
        self.mlp_dim= mlp_dim
        self.enc_layers=[TransformerBlock(embed_dim=d_model,mlp_dim=mlp_dim,num_heads=num_heads) for _ in range(num_layers)]
        self.mlp_head=tf.keras.Sequential([Dense(mlp_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                Dense(class_number),])
    def create_pos_encoding(self,d_model,num_patches):
        self.pos_emb=self.add_weight("pos_emb",(1,num_patches+1,d_model))
        self.class_emb=self.add_weight("class_emb",(1,1,d_model))
        return Dense(d_model)
    def create_patche(self,image_size,patch_size,channels):
        num_patches=(image_size//patch_size)**2
        self.patch_size=patch_size
        self.patch_dim=channels*patch_size**2
        return num_patches
    def extract_patches(self,images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches
    def call(self,images,training):
        batch_size=tf.shape(images)[0]
        images=self.rescale(images)
        patches=self.extract_patches(images)
        x=self.patch_proj(patches)
        class_embeding=tf.broadcast_to(self.class_emb,(batch_size,1,self.d_model))
        x=tf.concat([x,class_embeding],axis=1)
        x=x+self.pos_emb
        for layer in self.enc_layers:
            x=layer(x,training)
        x=self.mlp_head(x[:,0])
        return x
AUTOTUNE = tf.data.experimental.AUTOTUNE
# setting variables
IMAGE_SIZE=32
PATCH_SIZE=4
NUM_LAYERS=8
NUM_HEADS=16
MLP_DIM=128
lr=1e-3
WEIGHT_DECAY=1e-4
BATCH_SIZE=64
epochs=10
#Load the dataset
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()

model = VisionTransformer(
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    num_layers=NUM_LAYERS,
    class_number=10,
    d_model=64,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    channels=3,
    dropout=0.1,
)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=tfa.optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY),
    metrics=["accuracy"],
)
early_stop = tf.keras.callbacks.EarlyStopping(patience=10),
mcp = tf.keras.callbacks.ModelCheckpoint(filepath='weights/best.h5',
                                         save_best_only=True,
                                         monitor='val_loss',
                                         mode='min')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.1,
                                                 patience=3,
                                                 verbose=0,
                                                 mode='auto',
min_delta=0.0001, cooldown=0, min_lr=0)
model.fit(
    x_train,y_train,validation_data=[x_test,y_test],
    epochs=epochs
   #, callbacks=[early_stop, mcp, reduce_lr]
)
