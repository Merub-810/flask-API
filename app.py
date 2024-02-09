from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import csv

app = Flask(__name__)
version=0
dataset_file_path = "/Newfilev_2.csv"
model_file_path = "/Modelv_2.h5"
def process_and_predict(replace_word, category_file_path, Main_file_path,category_file_name):
    global version, dataset_file_path, model_file_path
    version= version+1
    try:
        New_file_path = f"C:\\Users\\DELL\\Model\\Newfilev_{version}.csv"
        with open(Main_file_path, 'r', newline='',encoding='utf-8') as Main_file:
            with open(New_file_path, 'w', newline='',encoding='utf-8') as New_file:
                # Create CSV reader and writer objects
                csv_reader = csv.reader(Main_file)
                csv_writer = csv.writer(New_file)

                # Copy the contents from the input file to the output file
                for row in csv_reader:
                    csv_writer.writerow(row)

        with open(category_file_path, 'r', encoding='utf-8') as category_file:
          file_content = category_file.read()
          if category_file_name == "کھانا":
             modified_content = file_content.replace("چاول", replace_word)
          elif category_file_name == "کپڑے":
              modified_content = file_content.replace("شوز", replace_word)
        #   elif category_file_name == "لوگ":
        #       modified_content = file_content.replace("شوز", replace_word)
          elif category_file_name == "جسمانی اعضا":
              modified_content = file_content.replace("ہاتھ", replace_word)
          elif category_file_name == "ٹرانسپورٹ":
              modified_content = file_content.replace("گاڑی", replace_word)
          elif category_file_name == "جانور":
              modified_content = file_content.replace("بلی", replace_word) 
          elif category_file_name == "چیزیں":
              modified_content = file_content.replace("فٹ بال", replace_word)   
          elif category_file_name == "جگہ":
              modified_content = file_content.replace("گھر", replace_word)   
        #   elif category_file_name == "سرگرمیاں":
        #       modified_content = file_content.replace("شوز", replace_word)    
        #   elif category_file_name == "احساسات":
        #       modified_content = file_content.replace("شوز", replace_word)          

        

                    
 
        with open(New_file_path, 'a', newline='', encoding='utf-8') as New_file:
        # csv_writer = csv.writer(output_file)
          New_file.write(modified_content)
          
    except FileNotFoundError:
        print(f"Error: File not found - {New_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

    data = pd.read_csv(New_file_path, encoding='utf-8')

    # Extract subjects, verbs, objects, and sentences from the DataFrame
    subjects = data['subjects'].fillna('').astype(str).tolist()
    verbs = data['verbs'].fillna('').astype(str).tolist()
    objects = data['objects'].fillna('').astype(str).tolist()
    sentences = data['sentences'].fillna('').astype(str).tolist()

    # Tokenize and prepare data for Urdu text
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(subjects + verbs + objects + sentences)
    vocab_size = len(tokenizer.word_index) + 1

    # Encode data
    subject_sequences = tokenizer.texts_to_sequences(subjects)
    verb_sequences = tokenizer.texts_to_sequences(verbs)
    object_sequences = tokenizer.texts_to_sequences(objects)
    sentence_sequences = tokenizer.texts_to_sequences(sentences)

    # Pad sequences to have the same length
    max_sequence_length = max(map(len, sentence_sequences))
    subject_sequences = tf.keras.preprocessing.sequence.pad_sequences(subject_sequences, maxlen=max_sequence_length, padding='post')
    verb_sequences = tf.keras.preprocessing.sequence.pad_sequences(verb_sequences, maxlen=max_sequence_length, padding='post')
    object_sequences = tf.keras.preprocessing.sequence.pad_sequences(object_sequences, maxlen=max_sequence_length, padding='post')
    sentence_sequences = tf.keras.preprocessing.sequence.pad_sequences(sentence_sequences, maxlen=max_sequence_length, padding='post')

    # Define input layers for subject, verb, and object
    input_subject = tf.keras.layers.Input(shape=(max_sequence_length,))
    input_verb = tf.keras.layers.Input(shape=(max_sequence_length,))
    input_object = tf.keras.layers.Input(shape=(max_sequence_length,))

    # Embedding layer shared across inputs
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=256, input_length=max_sequence_length)
    embedded_subject = embedding_layer(input_subject)
    embedded_verb = embedding_layer(input_verb)
    embedded_object = embedding_layer(input_object)

    # LSTM layers
    lstm_layer = tf.keras.layers.LSTM(700, return_sequences=True)
    encoded_subject = lstm_layer(embedded_subject)
    encoded_verb = lstm_layer(embedded_verb)
    encoded_object = lstm_layer(embedded_object)

    # Merge the encoded inputs
    merged = tf.keras.layers.Add()([encoded_subject, encoded_verb, encoded_object])

    # Decoder LSTM
    decoder_lstm = tf.keras.layers.LSTM(350, return_sequences=True)(merged)

    # Output layer
    output = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder_lstm)
    model = tf.keras.models.Model(inputs=[input_subject, input_verb, input_object], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Assuming you have a trained model saved as 'Modelt1.h5'
    # loaded_model = tf.keras.models.load_model('Modelt1.h5')
    model.fit(
    [subject_sequences, verb_sequences, object_sequences],
    tf.keras.utils.to_categorical(sentence_sequences, num_classes=vocab_size),
    epochs=10
    )
    model.save(f'Modelv_{version}.h5')
    model = tf.keras.models.load_model(f'Modelv_{version}.h5')
    # Example input for prediction
    input_subject = np.array(tokenizer.texts_to_sequences(['ماما']))
    input_verb = np.array(tokenizer.texts_to_sequences(['کھانا']))
    input_object = np.array(tokenizer.texts_to_sequences(['بریانی']))

    # Pad the input sequences
    input_seq = [input_subject, input_verb, input_object]
    input_seq = [tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_sequence_length, padding='post') for seq in input_seq]

    predicted_sequence = model.predict(input_seq)
    predicted_sentence = ' '.join([tokenizer.index_word[word_index] for word_index in np.argmax(predicted_sequence, axis=2)[0] if word_index in tokenizer.index_word])
    dataset_file_path = New_file_path
    # model_file_path = "C:\\Users\\DELL\\Model\\Modelv{version}.h5"
    model_file_path = "C:\\Users\\DELL\\Model\\Modelv_{}.h5".format(version)
    print(model_file_path)
    print(dataset_file_path)
    return predicted_sentence

@app.route('/trainAddNewWord', methods=['POST'])
def trainAddNewWord():
    try:
        # Get replace_word and category_file_name from the request
        data=request.json
        replace_word = data['card_name']
        category_file_name = data['category']
        print(replace_word)
        # Specify the path to your output CSV file
        # Main_file_path = "c:\\Users\\DELL\\Model\\Urdu1.csv"
        # New_file_path = "C:\\Users\\DELL\\Model\\Newfile.csv"
        # Load data from a CSV file
        category_file_path = f"c:\\Users\\DELL\\Model\\categories\\{category_file_name}.csv"
        print(category_file_path)
        # Process and predict
        predicted_sentence = process_and_predict(replace_word, category_file_path, dataset_file_path,category_file_name)

        return jsonify({"predicted_sentence": predicted_sentence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_sentence', methods=['POST'])
def predict_sentence():
    try:        
        model = tf.keras.models.load_model(model_file_path)
        print(model_file_path)
        # Load CSV file for vocabulary mapping (replace 'Urdu1.csv' with your file)
        data1 = pd.read_csv(dataset_file_path, encoding='utf-8')
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(data1['subjects'].fillna('').astype(str).tolist() +
                       data1['verbs'].fillna('').astype(str).tolist() +
                       data1['objects'].fillna('').astype(str).tolist() +
                       data1['sentences'].fillna('').astype(str).tolist())
        print(dataset_file_path)
         # Max sequence length
        max_sequence_length = max(map(len, tokenizer.texts_to_sequences(data1['sentences'].fillna('').astype(str).tolist())))
        # Create a list of input sequences
        data = request.json
        if not any(key in data for key in ['subject', 'verb', 'object']):
          return jsonify({'error': 'Please provide at least one of subject, verb, or object.'})
        input_subject = np.array(tokenizer.texts_to_sequences([data['subject']]))
        input_verb = np.array(tokenizer.texts_to_sequences([data['verb']]))
        input_object = np.array(tokenizer.texts_to_sequences([data['object']]))

        # Pad the input sequences
        input_seq = [input_subject, input_verb, input_object]
        input_seq = [tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_sequence_length, padding='post') for seq in input_seq]

        # Make predictions
        predicted_sequence = model.predict(input_seq)
        predicted_sentence = ' '.join([tokenizer.index_word[word_index] for word_index in np.argmax(predicted_sequence, axis=2)[0] if word_index in tokenizer.index_word])

        # Return the predicted sentence as JSON
        return jsonify({'predicted_sentence': predicted_sentence})

    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)
