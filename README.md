# Tensorflow2로 배우는 머신러닝 [융합 신기술 교육 프로그램2(AA0020)]

이 git 저장소는 인하대학교 공학교육혁신센터 **Tensorflow2로 배우는 머신러닝 [융합 신기술 교육 프로그램2(AA0020)]** 강의 및 실습 내용을 요약정리한 내용을 담고 있습니다.

## Index

1. [Basic](1_Basic)
    1. [Python](1_Basic/0_python.ipynb)
    2. [Numpy](1_Basic/1_numpy.ipynb)
    3. [Pandas](1_Basic/2_pandas.ipynb)

2. [KNN](2_KNN)

3. [Decision Tree](3_DecisionTree)

4. [Regression Analysis](4_Regression_Analysis)
    1. [Regression](4_Regression_Analysis/1_Regression.ipynb)
        - Linear Regression
        - Least Square Method
        - Logistic Regression
        - Cost Function / Sigmoid Function
        - Gradient Descent Method
    2. [Using Tensorflow](4_Regression_Analysis/2_Tensorflow.ipynb)
        - Linear Regression
        - Logistic Regression
        - MNIST Datasets
        - Categorical Encoding / Softmax Function
        - SGD / BGD
        - Epoch / Batch

5. [Neural Network](5_Neural_Network)
    1. [Neural Network](5_Neural_Network/1_NeuralNetwork.ipynb)
        - Neuron Cell / Perceptron
        - Neural Network
        - Hidden Layer
        - Backpropagation
    2. [Using TensorFlow](5_Neural_Network/2_Tensorflow.ipynb)
        - Gradient Vanishing Problem
        - ReLU / LeakyReLU / ELU Function
        - Optimizers
            - Xavier / He Initialization
            - Dropout / Batch Normalization
            - Momentum / Nesterov Momentum
            - AdaGrad / RMSProp / Adam

6. [Convolutional Neural Network](6_CNN)
    1. [CNN](6_CNN/1_CNN.ipynb)
        - Convolutional Layer
        - Pooling Layer
        - Toy Image
    2. [MNIST CNN](6_CNN/2_MNIST_CNN_TF2.ipynb)
    3. [MNIST CNN DNN](6_CNN/3_MNIST_CNNDNN_TF2.ipynb)
    4. [Example: Fashion MNIST](6_CNN/4_Fashion_MNIST.ipynb)
    5. [Example: CIFAR10 Dataset](6_CNN/5_CIFAR10)
        1. [CIFAR10 w/ Dropout (approx. 74%)](6_CNN/5_CIFAR10/1_CIFAR10_74.ipynb)
        2. [CIFAR10 w/ Dropout, Batch Normalization (approx. 79%)](6_CNN/5_CIFAR10/2_Batch_Normalization_79.ipynb)
        3. [CIFAR10 w/ Dropout, Batch Normalization, Kernel Regularizer (approx. 84%)](6_CNN/5_CIFAR10/3_Kernel_Regularization_84.ipynb)
        4. [CIFAR10 w/ Dropout, Batch Normalization, Kernel Regularizer, ImageDataGenerator (approx. 88%)](6_CNN/5_CIFAR10/4_ImageDataGenerator_88.ipynb)
    6. [Example: CatsVsDogs Dataset](6_CNN/6_Cat_Dog)

7. [Auto Encoder](7_AutoEncoder)

8. [Recurrent Neural Network](8_RNN)
    1. [RNN](8_RNN/1_RNN.ipynb)
        - Time Sequence Forecasting
        - RNN
            - one to one
            - one to many
            - many to one
            - many to many
            - Multiple Layer RNN
        - Example: Character RNN
        - TimeDistributed Layer
        - Embedding Layer
        - Projection Layer
    2. [RNN](8_RNN/2_RNN.ipynb)
        - Gradient Vanishing in RNN
        - Long Short Term Memory
        - Stacked RNN
    3. [Example: Stock Data](8_RNN/3_StockData.ipynb)
    4. [Example: IMDB Dataset](8_RNN/4_IMDB.ipynb)
    5. [Word2Vec](8_RNN/5_Word2Vec.ipynb)
    6. [Seq2Seq](8_RNN/6_Seq2Seq.md)
        - Attention

9. [Appendix](9_Appendix)
    1. VAE
    2. Image / Video Preprocessing