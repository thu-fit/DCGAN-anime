from model import DCGAN
import tensorflow as tf
from utils import *
from ops import *
import numpy as np

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
