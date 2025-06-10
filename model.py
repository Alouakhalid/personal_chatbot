import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
import random
import json
from tqdm import tqdm

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open('data/intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignor_words = ['?', '!', '.', ',']

for intent in tqdm(intents['intents'], desc="Processing intents"):
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignor_words]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('data/words.pkl', 'wb'))
pickle.dump(classes, open('data/classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in tqdm(documents, desc="Building training data"):
    bag = []
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in document[0] if w not in ignor_words]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

x_train = np.array(list(training[:, 0]))
y_train = np.array(list(training[:, 1]))

model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=1)
loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
print(f"Model Loss: {loss:.4f}, Model Accuracy: {accuracy:.4f}")
model.save('data/chatbot_model.h5',hist)
print("✅ Model trained and saved successfully.")
