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
Link to work: https://github.com/constantinembufung/Word-Embeddings

