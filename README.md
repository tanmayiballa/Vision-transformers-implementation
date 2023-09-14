# Vision-transformers-implementation
Implementation of Vision Transformers from scratch.

**Abstract** 

Vision Transformers (ViTs) have been the major focus in the field of computer vision after they have demonstrated promising results in natural language processing. In this project, we have implemented a vision transformer using PyTorch for image classification, with a focus on efficiency and reduced complexity. The motivation behind ViTs is the need for object recognition models that can handle variable-sized inputs and model long-range dependencies. While CNNs have been successful in this field, we explored the potential of vision transformers and aimed to reduce their computational overhead. The focus of this project is on the following research question: Can we reduce the complexity of the model while maintaining decent accuracy? Our base model consists of minimal encoder blocks and a reduced number of parameters, which enabled us to perform training using little computational complexity. We were able to achieve good results with just 15-20 training epochs. We tested our model on different types of classification tasks that include datasets such as CIFAR10, MNIST, and Satellite Imagery. The performance of our base model on MNIST and Satellite Imagery has been significant with an accuracy of about 91% on MNIST and 85% on Satellite Imagery. Additionally, we performed hyperparameter tuning to further improve the efficiency of the base model. We further tried various techniques such as augmentation, etc., and commented on a few findings regarding the same. In a nutshell, we have demonstrated that a simplified vision transformer can still achieve good results in image classification tasks for simpler datasets.

**Model Architecture**

![Flowchart of the model](https://github.com/tanmayiballa/Vision-transformers-implementation/blob/main/Materials/Flowchart.png)

**Sample Image Patching**

![Alt text](https://github.com/tanmayiballa/Vision-transformers-implementation/blob/main/Materials/Image_patching.jpg)

**Implementation**

We have implemented our base model using PyTorch. The input images (H*W*C) are divided into non-overlapping patches with a patch size of 4 (P*P*C); flattened into a 1D sequence of tokens; and linearly projected to the lower dimensional space. nn.linear function in PyTorch is used for these projections. The positional encodings are added to these linear embeddings along with the classification token. Further, these embeddings are passed through the layer normalization of the encoder block. We’ve used nn.LayerNorm function in PyTorch for this and passed through the self-attention layer which is carried out via PyTorch’s nn.MultiheadAttention function. These are further processed through the fully connected layers and MLP Classifier block, both of which were created using nn.Linear function from PyTorch.

**PyTorch model**


![Flowchart of the model](https://github.com/tanmayiballa/Vision-transformers-implementation/blob/main/Materials/model.png))

**Model Training**


The training is performed using nn.module class in Pytorch. This module is popular for creating deep learning models since it automatically computes the gradients as the autograd system in Pytorch. We have created a custom class of nn.module, defined the optimizers and loss functions, and implemented the transformer architecture in the forward pass of nn.module.
 
We have used an Adam optimizer for effective convergence of the model, which is a gradient- descent based optimizer, that is known for constantly updating the learning rate for each parameter of the model, based on the gradients. This adaptive learning rate helps the model in efficient convergence. Cross-entropy loss function is used during the training. The parameters of our base model are clearly depicted in Table 1.

**Table 1**
| Parameter name  | Parameter value |
| -------  | - |
| Patch-size  | 4 |
| Batch-size | 32 |
| Embed dimensions  | 128  |
| Hidden dimensions  | 256  |
| Encoder blocks  | 2  |
| n_heads in encoder  | 4  |
| Loss  | Cross Entropy Loss  |


**Experimental Results**

_Dataset:_
We considered three different data sets.
1. **MNIST (single channel)** - This dataset consists of 70000 grayscale images of handwritten digits 0 to 9, each of size 28 x 28 pixels. The images are divided into two subsets-a training set of 60,000 images and
a test set of 10,000 images.
2. **CIFAR10 (Natural task)** - This dataset
consists of 60,0000 color images in 10 classes, with 6000 images per class. The images are of size 32x32 pixels and are split into a training set of 50,000 images and a test set of 10,000 images.
3. **Satellite Imagery (specialized task)** - This dataset consists of 12,000 color images in 2 classes of damage vs. no_damage with each of 228 x 228 pixels. The images are divided into two subsets of training set of 10,000 images with 5,000 images in each class and a test set of 2,000 images with 1,000 images in each class.

_Key Metrics Used:_
1. **Accuracy**: Percentage of correct
predictions by the model.
2. **Top2 Accuracy**: Top2 accuracy considers
the prediction as the correct prediction, even if the model has computed the second highest probability for the correct label.
3. **Top3 Accuracy**: Top3 accuracy considers the prediction as the correct prediction, even if the model has computed the third-highest probability for the correct label.
4. The top2 and top3 accuracies are mentioned along with the mean difference of the second/third-highest probability with the maximum probability.

_Results & Discussions:_

![Alt text](https://github.com/tanmayiballa/Vision-transformers-implementation/blob/main/Materials/results.jpg)

**Table 2**

| Model	| Epochs | Total Accuracy (%) | Top2 accuracy % (with avg. difference from top1) | Top2 accuracy % (with avg. difference from top1)|
| ---- | ----- | ------------------ | ---------------------------------------------- | ------------------------------------------------|
| MNIST | 10	| 90.28 | 96.89(0.42)|	98.62(0.44) |
| MNIST | 15	| 91.18	| 96.92(0.44)|	98.61(0.45) | 
| MNIST | 20	| 91.47 | 97.57(0.43)|	99.09(0.46) |
| Hurricane damage | 10	| 84.24 | NA |	NA |
| Hurricane damage | 15	| 85.98	| NA |	NA | 
| Hurricane damage | 20	| 85.87 | NA |	NA |
| CIFAR10 | 10	| 42.32 | 62.48(0.19)|	74.29(0.21) |
| CIFAR10 | 15	| 43.43	| 63.32(0.2)|	75.18(0.24) | 
| CIFAR10 | 20	| 43.78 | 63.30(0.24)|	75.32(0.27) |


As depicted in the table above, the performance of our model on MNIST and Hurricane Damage dataset has been decent. The top2 accuracy of MNIST has reached around **97% for 20 epochs** with a mean difference of 0.43. The effective background foreground distinction in the MNIST dataset might also be a major boost for the model performance. The training and testing accuracies of MNIST and CIFAR10 throughout the training are depicted in Fig.4. and Fig.5. Though the model was able to generalize well for the MNIST dataset, it seemed to overfit on CIFAR10. The same is discussed in the gradient saturation for CIFAR10, in the later part of this section.

However, the model failed to perform well on CIFAR10 dataset. Hence, we have tried to implement a few techniques to improve the model accuracy, which are listed below.

_Experiments performed:_
1. Image augmentation: Since our model
didn’t perform well for CIFAR10 dataset, we tried using image augmentation techniques that include horizontal flip, vertical flip, resize, and random crop. However, these techniques didn’t provide enough information for the model to achieve good accuracy. The low resolution (32*32*3) of the images in CIFAR10 dataset might also be the reason for the same.
2. Gradient Saturation for CIFAR10: After a few epochs, the test loss of the CIFAR10 dataset, started to increase, though the training loss kept decreasing. This clearly depicts that the model has started overfitting on the training dataset. Hence, we tried techniques such as dropout, to randomly remove a few layers during training, to avoid overfitting. However, dropout techniques couldn’t improve the accuracy of CIFAR10 significantly.
3. Patch size of the model also plays an important role in the performance. A model with a larger patch size will be able to efficiently capture global dependencies in the input sequence but might fail to capture local spatial information. Larger patch sizes are preferred for datasets that have clear backgrounds and high-resolution images. However, since CIFAR10 datasets have low-resolution images, a larger patch size wouldn’t be ideal for the same. Hence, we have considered a patch size of 4 which performed well as compared to a patch- size of 8.

**Key Findings:**
1.	For single channel and specialized classification tasks, using complex models with lots of layers and parameters is not required. A simple Vit model with fewer layers and parameters is sufficient to obtain decent accuracy with just 10-20 epochs.
2.	More layers and parameters can improve image classification accuracy for natural tasks (CIFAR10), but minimal computation can still provide decent top2 & top3 accuracy. This model is suitable when some ambiguity is allowed in the task and requires less computational power.
3.	Image augmentation techniques are effective for many datasets, but they might not have much impact on datasets that contain diverse and many images. Our experiments with CIFAR10 have shown this to be the case. Owing to this diversity, using a model with a lower patch size yielded better results by capturing finer details in the patches.
4.	Gradient Saturation for CIFAR10 model suggested that the model is not able to capture any more information. This can also be the motivation to include many encoder blocks and layers in the architecture for complex datasets.
