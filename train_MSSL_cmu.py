import os.path
import numpy as np
import tensorflow as tf
import cv2
from nets import MSSL_cmu as MSSL
from data_provider import dataloader
from utils import preprocess
from utils import metrics
from utils import optimizer
from utils import recovercmu_3d
import time
import scipy.io as io
import os,shutil
import pdb
# -----------------------------------------------------------------------------
FLAGS = tf.app.flags.FLAGS

# data I/O
tf.app.flags.DEFINE_string('dataset_name', 'skeleton',
                           'The name of dataset.')
tf.app.flags.DEFINE_string('train_data_paths',
                           'data/h36m20/h36m20_train_3d.npy',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                           'data/h36m20/h36m20_val_3d.npy',
                           'validation data paths.')
tf.app.flags.DEFINE_string('test_data_paths',
                           'data/h36m20/test20_npy',
                           'test data paths.')
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/h36m',
                            'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_dir', 'results/h36m',
                           'dir to store result.')
# model
tf.app.flags.DEFINE_string('model_name', 'SDMTL',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', '',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 10,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 20,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('jointsdim', 3,
                            'Joint dimension.')
tf.app.flags.DEFINE_integer('jointsnum', 25,
                            'Joint number to predict.')
tf.app.flags.DEFINE_integer('totaljointsnum', 38,
                            'total joint number of the human body.')
tf.app.flags.DEFINE_integer('channel', 1,
                            'number of image channel.')
tf.app.flags.DEFINE_integer('filter_size', 3,
                            'filter of a cascade multiplicative unit.')
tf.app.flags.DEFINE_integer('Tu_length', 1,
                            'The number of SU in TU blocks.')

tf.app.flags.DEFINE_string('num_hidden', '64',
                           'number of units in a cascade multiplicative unit.')
tf.app.flags.DEFINE_float('min_err', 100,
                           'predefine minimize errors.')

# optimization
tf.app.flags.DEFINE_float('lr', 0.001,
                          'base learning rate.')
tf.app.flags.DEFINE_boolean('reverse_input', True,
                            'whether to reverse the input frames while training.')
tf.app.flags.DEFINE_integer('batch_size', 8,
                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 80000,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 20,
                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 10000,
                            'number of iters saving models.')
tf.app.flags.DEFINE_integer('num_save_samples', 30,
                            'number of sequences to be saved.')
tf.app.flags.DEFINE_integer('n_gpu', 4,
                            'how many GPUs to distribute the training across.')

# extra parameters of encoder and decoder
tf.app.flags.DEFINE_integer('encoder_length', 4,
                            'number of encoder residual multiplicative block of predCNN')
tf.app.flags.DEFINE_integer('decoder_length', 6,
                            'number of decoder residual multiplicative block of predCNN')


class Model(object):
    def __init__(self):
        # inputs
        self.x = [tf.placeholder(tf.float32,
                                 [FLAGS.batch_size,
                                  FLAGS.seq_length,
                                  FLAGS.jointsnum,
                                  FLAGS.jointsdim,
                                  FLAGS.channel])
                  for i in range(FLAGS.n_gpu)]


        grads = []
        loss_train = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        self.params = dict()
        if 'SDMTL' in FLAGS.model_name:
            self.params['encoder_length'] = FLAGS.encoder_length
            self.params['decoder_length'] = FLAGS.decoder_length
        num_hidden = [int(x) for x in FLAGS.num_hidden.split(',')]
        for i in range(FLAGS.n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(tf.get_variable_scope(),
                                       reuse=True if i > 0 else None):
                    # define a model
                    output_list = SDMTL.SDMTL(
                        self.x[i],
                        self.params,
                        num_hidden,
                        FLAGS.filter_size,
                        FLAGS.seq_length,
                        FLAGS.input_length,
                        FLAGS.Tu_length)                  

                    gen_ims = output_list[0]
                    loss = output_list[1]
                    pred_ims = gen_ims[:, FLAGS.input_length - FLAGS.seq_length:]
                    loss_train.append(loss / FLAGS.batch_size)
                    # gradients
                    all_params = tf.trainable_variables()
                    grads.append(tf.gradients(loss, all_params))
                    self.pred_seq.append(pred_ims)

        if FLAGS.n_gpu == 1:
            self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
        else:
            # add losses and gradients together and get training updates
            with tf.device('/gpu:0'):
                for i in range(1, FLAGS.n_gpu):
                    loss_train[0] += loss_train[i]
                    for j in range(len(grads[0])):
                        grads[0][j] += grads[i][j]
            # keep track of moving average
            ema = tf.train.ExponentialMovingAverage(decay=0.9995)
            maintain_averages_op = tf.group(ema.apply(all_params))
            self.train_op = tf.group(optimizer.adam_updates(
                all_params, grads[0], lr=self.tf_lr, mom1=0.95, mom2=0.9995),
                maintain_averages_op)

        self.loss_train = loss_train[0] / FLAGS.n_gpu

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)
        self.sess.run(init)
        if FLAGS.pretrained_model:
            self.saver.restore(self.sess, FLAGS.pretrained_model)
            print 'pretrain model',FLAGS.pretrained_model

    def train(self, inputs, lr):
        feed_dict = {self.x[i]: inputs[i] for i in range(FLAGS.n_gpu)}
        feed_dict.update({self.tf_lr: lr})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

    def test(self, inputs):
        feed_dict = {self.x[i]: inputs[i] for i in range(FLAGS.n_gpu)}
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir)


def main(argv=None):
    if ~tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.MakeDirs(FLAGS.save_dir)
    if ~tf.gfile.Exists(FLAGS.gen_dir):
        tf.gfile.MakeDirs(FLAGS.gen_dir)

    print 'start training !',time.strftime('%Y-%m-%d %H:%M:%S\n\n\n',time.localtime(time.time()))
    # load data
    train_input_handle, test_input_handle = dataloader.dataloader(
        FLAGS.train_data_paths, FLAGS.valid_data_paths,
        FLAGS.batch_size * FLAGS.n_gpu, FLAGS.jointsdim,FLAGS.totaljointsnum, FLAGS.channel, FLAGS.input_length,FLAGS.seq_length,is_training=True)

    print('Initializing models')

    model = Model()
    lr = FLAGS.lr
    train_time=0
    test_time_all=0
   
    min_err=FLAGS.min_err
    errlist=[]
    for itr in range(1, FLAGS.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        start_time = time.time()
        ims = train_input_handle.get_batch()
        ims = ims[:,:,0:FLAGS.jointsnum,:]
        ims_list = np.split(ims, FLAGS.n_gpu)

        '''
        if itr % 20000 == 0 and itr>0.000001: # and itr > 2000:
            lr = lr *0.96
        '''

        cost = model.train(ims_list, lr)

        if FLAGS.reverse_input:
            ims_rev = np.split(ims[:, ::-1], FLAGS.n_gpu)
            cost += model.train(ims_rev, lr)
            cost = cost/2
        end_time = time.time()
        t=end_time-start_time
        train_time += t

        if itr % FLAGS.display_interval == 0:
            print('itr: ' + str(itr)+' lr: '+str(lr)+' training loss: ' + str(cost))

        if itr % FLAGS.test_interval == 0:
            print('train time:'+ str(train_time))
            print('test...')
            str1 = 'basketball','basketball_signal','directing_traffic','jumping','running','soccer','walking','washwindow'
            res_path = os.path.join(FLAGS.gen_dir, str(itr))
            if  not tf.gfile.Exists(res_path):
                os.mkdir(res_path)
            batch_id = 0
            test_time=0
            mpjpe = np.zeros([1,FLAGS.seq_length - FLAGS.input_length])
            mpjpe_l = np.zeros([1,FLAGS.seq_length - FLAGS.input_length])
            f = 0
            for s in str1:
                start_time1 = time.time()
                batch_id = batch_id + 1
                mpjpe1=np.zeros([1,FLAGS.seq_length - FLAGS.input_length])
                tem = np.load(FLAGS.test_data_paths+'/test_cmu_'+str(FLAGS.seq_length)+'_'+s+'.npy')
                tem=np.repeat(tem,(FLAGS.batch_size*FLAGS.n_gpu)/8,axis=0)
                test_ims =tem[:,0:FLAGS.seq_length,:,:]
                test_ims1= test_ims
                test_ims=test_ims[:,:,0:FLAGS.jointsnum,:]
                test_dat = np.expand_dims(test_ims, axis=-1)
                test_dat = np.split(test_dat, FLAGS.n_gpu)
                img_gen = model.test(test_dat)
                end_time1 = time.time()
                t1=end_time1-start_time1
                test_time += t1
                # concat outputs of different gpus along batch
                img_gen = np.concatenate(img_gen)
                img_gen = img_gen[:,:,:,:,0]
                gt_frm = test_ims1[:,FLAGS.input_length:]
                img_gen = recovercmu_3d.recovercmu_3d(gt_frm,img_gen)

                # Mpjpe
                for i in range(FLAGS.seq_length - FLAGS.input_length):
                    x = gt_frm[:, i , :, ]
                    gx = img_gen[:, i, :, ]
                    for j in range(FLAGS.batch_size * FLAGS.n_gpu):
                        tem1=0
                        for k in range(img_gen.shape[2]):
                            tem1 += np.sqrt(np.square(x[j,k] - gx[j,k]).sum())
                        mpjpe1[0,i] += tem1/img_gen.shape[2]

                # save prediction examples
                '''
                path = os.path.join(res_path, str(batch_id))
                if  not tf.gfile.Exists(path):
                    os.mkdir(path)
                for i in range(FLAGS.seq_length):
                    name = 'gt' + str(i+1) + '.mat'
                    file_name = os.path.join(path, name)
                    img_gt = test_ims[0, i, :, :]
                    io.savemat(file_name, {'joint': img_gt})
                for i in range(FLAGS.seq_length-FLAGS.input_length):
                    name = 'pd' + str(i+1+FLAGS.input_length) + '.mat'
                    file_name = os.path.join(path, name)
                    img_pd = img_gen[0, i, :, :]
                    io.savemat(file_name, {'joint': img_pd})
                '''
                mpjpe1 = mpjpe1/(FLAGS.batch_size * FLAGS.n_gpu)
                print 'current action mpjpe: ',s
                for i in mpjpe1[0]:
                    print i
                mpjpe +=mpjpe1
                if f<=3:
                    print 'four actions',s
                    mpjpe_l += mpjpe1
                f=f+1
            test_time_all += test_time
            mpjpe=mpjpe/(batch_id)
            print( 'mean per joints position error: '+str(np.mean(mpjpe)))
            current_err=np.mean(mpjpe)
            errlist.append(current_err)
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(mpjpe[0,i])
            mpjpe_l=mpjpe_l/4
            print('mean mpjpe for four actions: '+str(np.mean(mpjpe_l)))
            for i in range(FLAGS.seq_length - FLAGS.input_length):
                print(mpjpe_l[0,i])
            print 'current test time:'+str(test_time)
            print 'all test time: '+str(test_time_all)
            filename = os.path.join(res_path, 'test_result')
            io.savemat(filename, {'mpjpe':mpjpe})
        if itr % FLAGS.snapshot_interval == 0 and min(errlist) < min_err:
            model.save(itr)
            min_err = min(errlist)
            print 'model saving done! ', time.strftime('%Y-%m-%d %H:%M:%S\n\n\n',time.localtime(time.time()))
        if itr % FLAGS.snapshot_interval == 0:
            print 'min mpjpe error:',min_err
        '''
        if itr % (5*FLAGS.snapshot_interval)==0:
            bakfile=path_bak+'/'+str(folder)
            shutil.copytree(FLAGS.save_dir,bakfile)
            folder=folder+1
		'''
        train_input_handle.next()


if __name__ == '__main__':
    tf.app.run()

