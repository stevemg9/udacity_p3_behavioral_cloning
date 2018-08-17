#**Behavioral Cloning**
### Result Video
[Behavioral Cloning](https://youtu.be/BWB_7BgosdA)

(Sorry about the low resolution!)

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For this project, I implemented the NVidia network discussed in the lectures but very slightly modified it.  My structure was as follows:

Preprocessing:
	-Normalize image pixel values between -0.5 and +0.5
	-Crop out top 70 and bottom 25 rows of pixels to eliminate unnecessary scene data and the front hood of the vehicle
CNN:
- Layer 1: Convolutional
	- Filter Size: 24x5x5
	- Stride: 2x2
	- Activation: relu
	- Regularization: L2
- Layer 2: Convolutional
	- Filter Size: 36x5x5
	- Stride: 2x2
	- Activation: relu
	- Regularization: L2
- Layer 3: Convolutional
	- Filter Size: 48x5x5
	- Stride: 2x2
	- Activation: relu
	- Regularization: L2
- Layer 4: Convolutional
	- Filter Size: 64x3x3
	- Stride: 1x1
	- Activation: relu
	- Regularization: L2
- Layer 5: Convolutional
	- Filter Size: 64x3x3
	- Stride: 1x1
	- Activation: relu
	- Regularization: L2
- Layer 6: Flatten
- Layer 7: Fully Connected
	- Size: 100x1
	- Activation: relu
	- Regularization: L2
- Layer 8: Fully Connected
	- Size: 50x1
	- Activation: relu
	- Regularization: L2
- Layer 9: Fully Connected
	- Size: 10x1
	- Activation: relu
	- Regularization: L2
- Layer 10: Output
	- Size: 1x1

I also experimented with different activation functions, dropout, and L2 regularization.  In the end I chose L2 regularization over dropout and relu  activations while training for 5 epochs.

####2. Attempts to reduce overfitting in the model

In order to reduce overfitting in the model, I used L2 regularization in th model.  I applied L2 regularization to all convolutional and fully connected layers.  L2 regularization will penalize large weights, essentially forcing the model to be dependent on more features by encouraging many smaller valued weights vs fewer larger ones.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.  I trained for 5 epochs, and mirrored all of the data as to not
bias the model to right or left turns.

####4. Appropriate training data

It took quite a while for me to find the right kind of data to train a successful model.  I tried including data from both tracks and using the left and right offset cameras, but in the end, both of these proved to cause problems.  I found that tuning the steering angles for the left/right offset was difficult to tune and it was, in fact, easier to record "recovery" data.  Also, when I included data from both tracks in order to generalize, the model became rather erratic and I found it was much more feasible to train a model to drive only on one track.  I wrote a separate script in oder to visualize the data.  There is a saved PNG called "visualize_data.png" that randomly samples 5 images from the training set, lists their steering angle, and shows the corresponding cropped image below.

###Model Architecture and Training Strategy

####1. Solution Design Approach and Training Data

I started this project with the NVidia model and ended up keeping it's general architecture.  The model itself was very effective, but this project seemed to be much more about data selection than network architecture.  Once I normalized and cropped the data, and built out the NVidia model, the task became: "Find the correct data to train a successful network."  My first attempts were way off, I tried training the model on lots and lots of data from both tracks - both normal and recovery laps in both directions.  I was also mirroring the data and using both the right and left offset images.  Training on this data took a long time and yielded fairly poor results.  After lots of experimenting, I determined that I needed to train on less data, from only the track that the model will drive on. Once I did this, the results were significantly better. With a little bit more tuning and data manipulation I was able to train a successful model.

My final training data consisted of two clockwise and two counter-clockwise laps of track 1, along with 2 clockwise and 2 counter-clockwise "laps" of recovery maneuvers on track 1.

####2. Final Model Architecture

CNN:
	Layer 1: Convolutional
		-Filter Size: 24x5x5
		-Stride: 2x2
		-Activation: relu
		-Regularization: L2
	Layer 2: Convolutional
		-Filter Size: 36x5x5
		-Stride: 2x2
		-Activation: relu
		-Regularization: L2
	Layer 3: Convolutional
		-Filter Size: 48x5x5
		-Stride: 2x2
		-Activation: relu
		-Regularization: L2
	Layer 4: Convolutional
		-Filter Size: 64x3x3
		-Stride: 1x1
		-Activation: relu
		-Regularization: L2
	Layer 5: Convolutional
		-Filter Size: 64x3x3
		-Stride: 1x1
		-Activation: relu
		-Regularization: L2
	Layer 6: Flatten
	Layer 7: Fully Connected
		-Size: 100x1
		-Activation: relu
		-Regularization: L2
	Layer 8: Fully Connected
		-Size: 50x1
		-Activation: relu
		-Regularization: L2
	Layer 9: Fully Connected
		-Size: 10x1
		-Activation: relu
		-Regularization: L2
	Layer 10: Output
		-Size: 1x1
