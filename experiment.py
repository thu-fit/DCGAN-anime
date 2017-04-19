from model import DCGAN
import tensorflow as tf
from utils import *
from ops import *
import numpy as np
from utils_extended import *
import os

class Sketch2Image:
  def __init__(self, dcgan, FLAGS):
    self.dcgan = dcgan
    self.FLAGS = FLAGS
    self.output_height = FLAGS.output_height
    self.output_width = FLAGS.output_width

    self.f_dim = 64   # first layer feature dimension
    self.z_dim = dcgan.z_dim
    self.batch_size = dcgan.batch_size
    self.sess = dcgan.sess

    self.p_bn1 = batch_norm(name='p_bn1')
    self.p_bn2 = batch_norm(name='p_bn2')
    self.p_bn3 = batch_norm(name='p_bn3')


    #log folder
    self.logdir = "./projector_log"
    if not os.path.isdir(self.logdir):
      os.mkdir(self.logdir)

  def build_model(self):
    # z --> x　--> sketch, the pair(sketch, z) which will be used to train projector
    self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
    self.sketch = tf.placeholder(tf.float32, [self.batch_size, self.output_height, self.output_width, 1], name='sketch')

    self.x_generated = self.dcgan.sampler(self.z)
    
    # define projector  sketch --> z
    self.z_project = self.sketch2z(self.sketch)
    # loss of projector
    self.loss = tf.reduce_mean(tf.squared_difference(self.z, self.z_project))

    # projected x = G(z), used to compare with x_generated
    self.x_project = self.dcgan.sampler(self.z_project)

    # variables to train
    t_vars = tf.trainable_variables()
    self.p_vars = [var for var in t_vars if 'p_' in var.name]
    
    # define summaries, which can be shown by tensorboard
    loss_sum = scalar_summary("p_loss", self.loss)
    z_sum = histogram_summary("z", self.z)
    z_project_sum = histogram_summary("z_project", self.z_project)
    x_generated_sum = image_summary("x_generated", self.x_generated)
    sketch_sum = image_summary("sketch", self.sketch)
    x_project_sum = image_summary("x_project", self.x_project)
    self.sum_merged = merge_summary([loss_sum, z_sum, z_project_sum, 
                                      x_generated_sum, sketch_sum, x_project_sum])
    self.writer = SummaryWriter(self.logdir, self.sess.graph)


  def train(self, iteration):
      # optimizer
      self.optim = tf.train.AdamOptimizer(self.FLAGS.learning_rate, beta1=self.FLAGS.beta1) \
              .minimize(self.loss, var_list = self.p_vars)

      # initialize
      try:
        tf.global_variables_initializer().run()
      except:
        tf.initialize_all_variables().run()

      # load model
      could_load, checkpoint_counter = self.dcgan.load(self.dcgan.checkpoint_dir)
      if could_load:
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
      else:
        print(" [!] Load failed...")

      # gao_shi_qing
      for it in xrange(iteration):
        # generate a pair of batch samples (sketch, z)
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
              .astype(np.float32)
        batch_x, = self.sess.run([self.x_generated], feed_dict = {self.z: batch_z})
        # print(self.sess.run([tf.shape(self.x_generated)]))
        # print(np.shape(batch_x))
        batch_sketch = image2edge(batch_x).astype(np.float32)

        # train the projector using the generated pair (sketch, z)
        _, loss_, _, summary_str = self.sess.run([self.optim, self.loss, self.x_project, self.sum_merged], 
          feed_dict = {
            self.sketch: batch_sketch,
            self.z: batch_z
          })
        self.writer.add_summary(summary_str, it)

        print("iteration: {}, loss: {} ".format(it, loss_))


  def sketch2z(self, sketch, batch_size=None, reuse=False):
    '''construct graph which maps a sketch to z
    '''

    if batch_size is None:
      batch_size = self.batch_size

    with tf.variable_scope("sketch2z") as scope:
      if reuse:
        scope.reuse_variables()
      
      h0 = lrelu(conv2d(sketch, self.f_dim, name='p_h0_conv'))
      h1 = lrelu(self.p_bn1(conv2d(h0, self.f_dim*2, name='p_h1_conv')))
      h2 = lrelu(self.p_bn2(conv2d(h1, self.f_dim*4, name='p_h2_conv')))
      h3 = lrelu(self.p_bn3(conv2d(h2, self.f_dim*8, name='p_h3_conv')))
      z = linear(tf.reshape(h3, [batch_size, -1]), self.z_dim, 'p_h3_lin')

    return tf.nn.tanh(z)








def project_x_to_z(dcgan, iteration, sess, FLAGS):
    # load an image
    image_path = './data/face/1.jpg'
    x = [ get_image(image_path,
              input_height=360,
              input_width=360,
              resize_height=FLAGS.output_height,
              resize_width=FLAGS.output_width,
              is_crop=True,
              is_grayscale=False)]

    # variables
    x = np.array(x).astype(np.float32)
    x = tf.Variable(x, tf.float32, name='x')
    # print("shape:-------",tf.shape(x))
    
    z_project = np.random.uniform(-1, 1, size=(1 , dcgan.z_dim)).astype(np.float32)
    z_project = tf.Variable(z_project, tf.float32, name='z_project')


    x_generated = dcgan.sampler(z_project, batch_size=1)


    # loss: mean square error
    recstr_loss = tf.reduce_mean(tf.squared_difference(x, x_generated))

    z_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
              .minimize(recstr_loss, var_list = [z_project])

    x_sum = image_summary("x", x)
    z_project_sum = histogram_summary("z_project", z_project)
    x_generated_sum = image_summary("generated_x", x_generated)
    loss_sum = scalar_summary("reconstruction_loss", recstr_loss)
    sum_merged = merge_summary([z_project_sum, x_sum, x_generated_sum, loss_sum])
    writer = SummaryWriter("./logs", dcgan.sess.graph)

    # saver = tf.train.Saver()
    # # initialize variables        
    # try:
    #   tf.global_variables_initializer().run()
    # except:
    #   tf.initialize_all_variables().run()

    # sess.run(tf.shape(x))
    # sess.run([init])
    # init = tf.variables_initializer([z_project, x])

    # initialize variables
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    # load model
    could_load, checkpoint_counter = dcgan.load(dcgan.checkpoint_dir)
    if could_load:
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for it in range(iteration):
        _ , loss_, summary_str = sess.run([z_optim, recstr_loss, sum_merged])
        # _, summary_str = self.sess.run([z_optim, train_z_sum])
        writer.add_summary(summary_str, it)

        # print loss
        if np.mod(it, 10) == 0:
          print("{}: MSE loss: {}".format(it, loss_))

        # if mod(it, 10) == 1:
        #   save_images(g_out, [1, 1], "./data/face/projected_{}.png".format(it))
