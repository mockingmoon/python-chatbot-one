# import nltk
import json
import pickle
import numpy as np
import random

#keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

wnl = WordNetLemmatizer()
# mystr = "There are my friends. They eat at hotels and restaurants."
# mystr2 = mystr.replace("."," ")
# mylist = [ wnl.lemmatize(val.lower()) for val in mystr2.split() ]
# print(mylist)

#initialization
words = []
docs = []
classes = []
ignore_words = ['?','!','.']

path2json = open("./intents.json").read()
intents = json.loads(path2json)

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word = word_tokenize(pattern)
        words.extend(word)
        docs.append((word, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# print(words)

words = [wnl.lemmatize(w.lower()) for w in words if w not in ignore_words]

words = sorted(list(set(words)))
print("Lemmatized tokens",len(words),"\n")

classes = sorted(list(set(classes)))
print("Tags classes", len(classes),"\n")

print("Combination of tags and patterns- ", len(docs),"\n")

# store above objects in pickle file .pkl
pickle.dump(words,open('words_stored.pkl','wb'))
pickle.dump(classes,open('classes_stored.pkl','wb'))

# generate bag of words i.e. 1 for word present, 0 for absent
# generate output list

training = []

for doc in docs:
    curr_words = [ wnl.lemmatize(wo.lower()) for wo in doc[0] if wo not in ignore_words ]
    bag = [ int(i in curr_words) for i in words ]
    output_list = [ int(tag==doc[1]) for tag in classes ]
    training.append([bag, output_list])

random.shuffle(training)
training = np.array(training)
x_train = list(training[:,0])
y_train = list(training[:,1])

# print("\n=====X-TRAIN=====\n")
# print(x_train)
# print("\n=====Y-TRAIN=====\n")
# print(y_train)
# print("\n==========\n")
print("Training data created.\n")

mymodel = Sequential()
mymodel.add(Dense(128, activation='relu', input_shape=(len(x_train[0]),)))
mymodel.add(Dropout(0.5))
mymodel.add(Dense(64, activation='relu'))
mymodel.add(Dropout(0.5))
mymodel.add(Dense(len(y_train[0]), activation='softmax'))

sgd = SGD(learning_rate=0.02,momentum=0.9,decay=1e-6,nesterov=True)
mymodel.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

# fitting and saving model
hist = mymodel.fit(x_train, y_train, batch_size=5, epochs=10, verbose=1)
mymodel.save('chatbot_mymodel.h5',hist)
print("Model created and saved!\n")