### Possible tasks
1. style transfer: people's face --> an anime(theoritically, this is feasible using GAN architecture)
2. generate under constrian: sketch of face --> an anime(this may be difficult, considering the huge feature difference between people's faces and anime)
3. modify the pictures according to user's edit

### method
+ dataset:
    * get dataset as indicated in <https://zhuanlan.zhihu.com/p/24767059>

+ GAN architecture:
    * conditional GAN
    * DCGAN
    * WGAN
    * BEGAN

+ for **task 1**: generate under constrain, sketch of face --> an anime:
    * solution 1:
        - train anime using GAN
        - find z to minimize the **similarity** of the sketch and the generated image
        - cons: it's hard to measure the similarity,
    * solution 2: train a network to map a sketch back to z space
        - for a z, generats an image x_g
        - use a filter to get a sketch of x_g, --> s
        - map s to z1 using a neural network P
        - train P to minimize distance between z and z1
        - **cons**: sketch to anime is one to many, while this solution is one to one, 
    * solution 3: using conditional GAN
        - for every iamge, generate a corresponding sketch
        - train a conditional GAN using the sketch as a condition 

+ for **task 2**style transfer, people's face --> an anime
    + solution 1: 
        + train face and anime together using GAN 
        + calculate the tanslation of this two classes in latent sapce v_t
        + for a given face, project it to latent space v_f, 
        + the generated anime will be: G(v_t + v_f)
    + solution 2:
        * extract the edage of face as a sketch, and transform this problem as goal 2

+ how to measure the similarity between two sketch?
    + use method in iGAN
    + using a discriminator

+ how to generate a sketch using an anime image?
    * 

+ how to find z to minimize a certain loss defined on the output of G?
    * directly use a optimizer to find optimal G(cons: generator is not convex, so may need many initial values)
    * in case of projection, train a network to map x to z
    
+ modify a generated pictures according to user's edit
    + use methods in iGAN


### plan
+ project a sketch to z
+ project a poeple's face to z using the boundary as constrain

### record:
##### for people's face, directly optimize z using MSE as loss
+ result very bad

##### using openCV to generate edges for each imgae
+ script get_image_edge.py
+ result is usable, but need to be improved

##### solution 2 of task 1
+ use trained generator to generate pairs (z, s). s is the sketch. 
    + pick z randomly
    + x = G(z)
    + z = cv2.Canny(s)
+ use supervised learning to train a map: s --> z
    * use network similar to discriminator
    * use euclidean distance in latent space as loss function

##### project an image to sketch

