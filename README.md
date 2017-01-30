# project3_behavioralcloning
self driving car nano degree Udacity 
I started this project by looking for a model that I could extract features from. After some research including reading the material on the forum and asking questions there, I came to the conclusion that because the images and the type of the problem are very different, feature extraction might not be the best option, although still a viable one. I picked commaio and NVidia models after testing both of them I noticed that NVidia model gave me a better start, meaning that the car in simulator stayed on the track for a longer stretch. I had to make some modification to this model such as adding the Keras BatchNormalization function as the normalization provided in NVidia code gave errors. Next layers are 5 convolutional layers the first layer has 24 layers of 5 by 5 filters. the second layer has 36 layers of 5 by 5. the third layer has 48 layers of 5 by 5. the fourth and fifth layer have 64 layers of 3 by 3 and they are followed by 4 fully connected layers. the activations are relu.
As for the type of data, I decided to use the dataset provided by Udacity and divided it into 80% train and 20% validation after shuffling. the whole dataset includes 8036 samples. the images are in a separate file. the driving_log_csv includes the addresses of the images and and steering angles in one row for each time stamp. I only used the image from center camera. The data generator function was used, because I had to create the batches on the fly, as putting the whole dataset in a data struction in memory would slow down the process, therefore batches of 128 was produced and two augmentation procedures were done inside the data generator function. one to shift the image and steering angle accordingly and the other to flip the image. These two functions are very similar to the funcions that were in this blog:https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.bax9cvv6n by Vivek Yadav.
