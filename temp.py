  def train_z(self, x, iteration):
    """Trains embedding z for an given image x using Adam optimizer"""

    # variables
    x = tf.Variable(x)
    x_sum = image_summary("x", x)
    
    z_project = tf.Variable(np.random.uniform(-1, 1, size=(1 , self.z_dim)))
    z_project_sum = histogram_summary("z_project", z_project)

    g_out = self.sampler(z_project)
    g_out_sum = image_summary("g_out", g_out)


    train_z_sum = [z_project_sum, x_sum, g_out_sum]

    # loss: mean square error
    recstr_loss = tf.reduce_mean(tf.squared_difference(x, g_out))

    z_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(recstr_loss, var_list = z_project)

    writer = SummaryWriter("./logs", self.sess.graph)
    # train
    for it in range(iteration):
      _, summary_str = self.sess.run([z_optim, train_z_sum])
      writer.add_summary(summary_str, it)

      # print loss
      err = recstr_loss.eval({})
      print("{}: MSE loss: {}".format(it, err))
      
      if mod(it, 10) == 1:
        save_images(g_out, [1, 1], "./data/face/projected_{}.png".format(it))


