import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Chargement des données
data = open('path_to_your_text_file.txt', 'r').read()

# Prétraitement des données
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
sequence_data = tokenizer.texts_to_sequences([data])[0]

vocab_size = len(tokenizer.word_index) + 1
seq_length = 50
sequences = []
for i in range(seq_length, len(sequence_data)):
    seq = sequence_data[i-seq_length:i+1]
    sequences.append(seq)

sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Construction du modèle
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Entraînement du modèle
model.fit(X, y, epochs=50, batch_size=256)

# Génération de texte
def generate_text(model, tokenizer, seq_length, seed_text, num_words):
    result = []
    in_text = seed_text
    for _ in range(num_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        y_pred = model.predict_classes(encoded, verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == y_pred:
                out_word = word
                break
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

seed_text = "Votre texte de départ"
generated = generate_text(model, tokenizer, seq_length, seed_text, 50)
print(generated)

