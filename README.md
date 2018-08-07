# -100DaysOfMLCode-OCR-Invoice-Recognition
In 100 days i will an OCR invoice recognition software 

## Day 0: July 11, 2018 
Today's Progress:  Learning Neural network and OpenCV .

Thoughts: Use NN to build a digit prediction model using the MNIST Dataset. final result of the model shown below, my model attend 98.2% accuracy.

Link to OpenCV: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html

Link to work: https://github.com/constantinembufung/Neural-Network-on-MNIST-dataset

## Day 1: July 12, 2018
Task: Learn word2vec with neural network
Today's Progress:  Learn word embedding in neural networks .

Thoughts: To store the word embeddings, we need a V*D matrix, V = vocabulary size, D = is the dimension (user-defined hyperparameter)

- downloaded several articles from Wikipedia of 61MB.
- read the data into a string, convert to lower case and tokenize it using nltk
- build a dictionary - maps string word to an ID eg. given the sentence I like to go to school
{I:0, like:1, to:2, go:3, school:4}
- reverse_dictionary: maps ID to string word e.g {O:I, 1:like, 2:to, 3:go, 4:school}
- count : list of list (word, frequency) e.g {(I,1), (like,1), (to,2), (go,1), (school,1)}
- data : contain string of text we read, string words are replaced with word IDs
- use UNK to represent unkown words that are not commonly used
Link to work: https://github.com/constantinembufung/Word-Embeddings

## Day 2, 3, 4, 5: July 13 - 16, 2018
Task: Learn Deep Learning with Keras
Today's Progress:  using keras to build a CNN for handwritten digits .

Thoughts: Understand backpropagation
- Build a CNN using the cifar10 datasets with 76.55% accuracy
- Learn how to do Data Augmentation when you have small training data
- Learn how to make your netword deep by adding more hidden layers, dropouts and using more CNN
- our model was like this Conv+Conv+maxpool+dropout+conv+conv+maxpool
Link to work: https://github.com/constantinembufung/deep_learning_with_keras

## Day 7: July 17, 2018
Tasks :  - Improving the CIFAR-10 performance with deeper a network: conv+conv+maxpool+dropout+conv+conv+maxpool
Followed by a standard dense+dropout+dense. All the activation functions are ReLU.
- Improving the CIFAR-10 performance with data augmentation
Thoughts: How to do data augmentation, doing predictions with real images, saving model as json
links: https://github.com/constantinembufung/deep_learning_with_keras


## Day 8, : July 18, 2018
Tasks :  - Image Classification using keras
- trains a neural network model to classify images of clothing, like sneakers and shirts
- I attend 93% in training but 88% in testing model. 
link: https://github.com/constantinembufung/deep_learning_with_keras/blob/master/Keras%20Basic%20image%20classifier%20-%20fashion%20.ipynb


## Day 9, : July 19, 2018
Tasks :  - Movies reviews using keras
- trains a neural network model to review movies as positive or negative
- I attend 93% in training but 87% in validation and testing model. 
link: https://github.com/constantinembufung/deep_learning_with_keras/blob/master/Movie%20reviews%20using%20keras.ipynb

## Day 10 : August 03, 2018 
Task: Test Generation using LSTM 
Learn: Today i learn how LSTM works and also learn word embedding 
Next task: Image captioning uusing LSTM link: @sirajraval

## Day 11: August 06, 2018 @100DaysOfMLCode
Day 11: learn Neural Machine Translation, understanding the maths behind LSTM in MT , study the BLEU score - evaluating the machine translation system. Encoder-Decoder architecture. 
Next Task: Build a seq2seq model for NMT from French-English using LSTM  