import tensorflow as tf
from layers.ResidualSpatialEncodingmodule import ResidualSpatialEncodingmodule as RSE
from layers.S2M import S2M as BSME 
import pdb
import numpy as np

def MSSL(images, params, num_hidden, filter_size, seq_length=20, input_length=10, Tu_length=1):
    encoder_length = params['encoder_length']
    decoder_length = params['decoder_length']
    num_hidden = num_hidden[0]
    channels = images.shape[-1]
    
    with tf.variable_scope('SDMTL'):
        # Encoder
        encoder_output = []
        #images1=images[:,:,:,:,0]
        images1=images
        for i in range(input_length):
            reuse = bool(encoder_output)
            ims = images1[:,i]
            input = Encoders(ims, num_hidden, filter_size, encoder_length, reuse)
            encoder_output.append(input)

        # Dynamicmodule ==> multi-grained trajectory modeling
        out1 = Dynamicmodule(encoder_output, num_hidden, filter_size, input_length, Tu_length, reuse=False)
        temp_out=out1[-1]
        
        weights = attention_temporal(out1[0], input_length, filter_size,reuse=False)
        temp_out1=attention_multiscale_temporal_fusion(temp_out,weights,num_hidden, filter_size,reuse=False)
        #temp_out1=multiscale_temporal_fusion(temp_out,num_hidden, filter_size,reuse=False)
        general_feat=out1[0]+temp_out1
        # Decoder
        output = []
        for i in range(seq_length - input_length):
            out = Decoders('Decoder_'+str(i), general_feat, num_hidden, filter_size, channels, decoder_length, reuse = False)
            output.append(out)
    # transpose output and compute loss
    gen_images = tf.stack(output)
    # [batch_size, seq_length, height, width, channels]
    gen_images = tf.transpose(gen_images, [1, 0, 2, 3, 4])
    gt_images = images[:, input_length:]
    #loss = tf.nn.l2_loss(gen_images - gt_images)
   # loss=tf.norm((gen_images-gt_images),ord=1)
    #loss = 100.0 * tf.reduce_mean(tf.norm(gen_images[:,:,:,:,0]-gt_images[:,:,:,:,0], axis=3, keep_dims=True, name='normal'))

    # attention loss
    pred_length=seq_length- input_length
    seqw=np.linspace(1,pred_length,pred_length,endpoint=True)
    seqw=np.exp(-0.3*seqw)
    seqw=seqw/np.sum(seqw)
    loss=0
    gen_images1=gen_images[:,:,:,:,0]
    gt_images = gt_images[:,:,:,:,0]
    for i in range(pred_length):
        loss += seqw[i] * tf.reduce_mean(tf.norm(gen_images1[:,i]-gt_images[:,i], axis=2, keep_dims=True, name='normal'))
    loss = 100*loss
    return [gen_images, loss]

def attention_temporal(x, input_length, filter_size,reuse):
    with tf.variable_scope('attention_weights', reuse=reuse):
        x=tf.layers.flatten(x)
        h = tf.layers.dense(inputs=x, units=300, activation=tf.sigmoid)
        h = tf.layers.dense(inputs=h, units=100, activation=tf.sigmoid)
        h = tf.layers.dense(inputs=h, units=(input_length-1), activation=tf.sigmoid)
        h = tf.reduce_mean(h, 0)
        return h

def attention_multiscale_temporal_fusion(x,weights,num_hidden, filter_size,reuse):
    with tf.variable_scope('multi_temporal_fusion', reuse=reuse):
        out=0
        for i in range(len(x)):
            weight=weights[i]
            temp=x[i]
            scale1=[]
            for j in range(len(temp)):
                if j == 0:
                    scale1=temp[j]
                else:
                    scale1 = tf.concat([scale1,temp[j]],axis=-1)
            scale1= tf.layers.conv2d(scale1, num_hidden, filter_size, padding='same', activation=tf.nn.leaky_relu,
                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                  name='fusion_layer_'+str(i))
            out = out+ weight*scale1
        return out
def multiscale_temporal_fusion(x,num_hidden, filter_size,reuse):
    with tf.variable_scope('multi_temporal_fusion', reuse=reuse):
        out=0
        for i in range(len(x)):
            temp=x[i]
            scale1=[]
            for j in range(len(temp)):
                if j == 0:
                   scale1=temp[j]
                else:
                   scale1 = tf.concat([scale1,temp[j]],axis=-1)
            scale1= tf.layers.conv2d(scale1, num_hidden, filter_size, padding='same', activation=tf.nn.leaky_relu,
                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                   name='fusion_layer_'+str(i))
            out = out+scale1
        return out

def Encoders(x, num_hidden, filter_size, encoder_length, reuse):
    with tf.variable_scope('Encoders', reuse=reuse):
        x = tf.layers.conv2d(x, num_hidden, 1, padding='same', activation=None,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='input_conv')
        for i in range(encoder_length):
            x = RSE('RSE_'+str(i+1), num_hidden // 2, filter_size)(x)
        return x


def Dynamicmodule(xs, num_hidden, filter_size, input_length, Tu_length, reuse):
    with tf.variable_scope('Multi_grained_trajectory', reuse=reuse):
        temp_out=[]
        xs0 = xs
        for i in range(10-1):  # old->input_length-1
            temp = []
            range_num = len(xs) // 2
            sep = len(xs) % 2
            for j in range(range_num):
                h1 = xs[2*j]
                h2 = xs[2*j+1]
                h = BSME('BSME_'+str(i+1), num_hidden, filter_size, Tu_length)(h1, h2, stride=False, reuse=bool(temp))
                idx = 2**(i+j+1)-1
                if idx>(len(xs0)-1):
                    idx=len(xs0)-1
                h = h + xs0[idx]
                temp.append(h)
            if sep == 1:
                h = xs[-1]
                h = BSME('BSME_'+str(i+1), num_hidden, filter_size, Tu_length)(h, h, stride=True, reuse=bool(temp))
                h = h +xs0[-1]
                temp.append(h)
            xs = temp
            temp_out.append(temp)
        return [xs[0],temp_out]


def Decoders(name, x, num_hidden, filter_size, output_channels, decoder_length, reuse):
    with tf.variable_scope('Decoders'+name, reuse=reuse):
        for i in range(decoder_length):
            x = RSE('RSE_'+str(i+1), num_hidden // 2, filter_size)(x)
        x = tf.layers.conv2d(x, output_channels, filter_size, padding='same',activation=None,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='output_conv')
        return x
