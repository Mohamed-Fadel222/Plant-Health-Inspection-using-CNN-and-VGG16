﻿# Plant-Health-Inspection-using-CNN-and-VGG16
### Architecture:
1. **Input Layer**: The input layer expects images of size 128x128 with 3 channels (RGB).
   
2. **Convolutional Layers**: The network consists of several convolutional layers, each followed by rectified linear unit (ReLU) activation functions. These layers aim to detect various features in the input images. The number of filters increases as the network goes deeper, which allows the network to capture more complex patterns.
   - The first convolutional layer has 32 filters with a kernel size of 3x3.
   - Subsequent convolutional layers follow a similar pattern, with increasing filter counts.
   - Padding is set to 'same' to ensure that the spatial dimensions of the feature maps remain the same after convolution.

3. **Max Pooling Layers**: After each pair of convolutional layers, a max-pooling layer is added. Max-pooling reduces the spatial dimensions of the feature maps, helping in reducing computation and controlling overfitting.
   - A pool size of 2x2 with a stride of 2 is used, which means the size of feature maps is halved after each max-pooling operation.

4. **Dropout Layers**: Dropout layers are added to prevent overfitting by randomly setting a fraction of input units to zero during training. This helps in improving the generalization ability of the model.
   - A dropout rate of 0.25 is applied after the last max-pooling layer.
   - Another dropout layer with a dropout rate of 0.4 is added before the output layer.

5. **Flattening Layer**: After the last max-pooling layer, a flattening layer is added to convert the 2D feature maps into a 1D vector, which can be fed into a fully connected layer.

6. **Fully Connected Layers**: Two dense (fully connected) layers are added for classification.
   - The first dense layer has 1500 units with ReLU activation.
   - Another dropout layer is added with a dropout rate of 0.4 to prevent overfitting.
   - The final dense layer has 38 units (assuming it's a multi-class classification task) with softmax activation, which outputs the probability distribution over the 38 classes.

