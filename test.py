from tensorflow.keras.preprocessing.text import Tokenizer
samples = ["grss is green and sun is hot"]
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
print(sequences)