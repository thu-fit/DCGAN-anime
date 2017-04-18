### goal
+ given a people's face, generate an anime
+ given a sketch of face, generate an anime
+ modify the pictures according to user's edit

### method
+ dataset:
    * get dataset as indicated in <https://zhuanlan.zhihu.com/p/24767059>
+ GAN architecture:
    * DCGAN
    * WGAN
    * BEGAN
+ projection a people's face to the dataset
    * directly optimize z use MSE as loss
    * extract boundary of people's face, then optimize z using this boundary as constrains.
+ given a sketch of face, generate an anime
    * use the sketch lines as constrains(as in iGAN)
+ how to optimize z?
    * directly use Adam on the generator to find z
    * train a network to map x to z(as in iGAN)
+ modify the pictures according to user's edit(as in iGAN)


### plan
+ for people's face, directly optimize z use MSE as loss
+ project a sketch to z
+ project a poeple's face to z using the boundary as constrain
+ 