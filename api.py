from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import base64

from flask import Flask, request, jsonify,render_template
import pickle


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, SpatialDropout1D, GlobalMaxPooling1D, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from flask_cors import CORS 

from tensorflow.keras.models import load_model
from joblib import load
from groq import Groq 
import logging

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
CORS(app)


# Paramètres
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
# Fonction pour construire les modèles individuels
def build_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))  # Modifier en fonction du nombre de classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_gru_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(GRU(128, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))  # Modifier en fonction du nombre de classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_cnn_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalMaxPooling1D())
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))  # Modifier en fonction du nombre de classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_bilstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(GlobalAveragePooling1D())
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))  # Modifier en fonction du nombre de classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Classe pour utiliser les modèles Keras dans scikit-learn
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn, epochs=10, batch_size=32, validation_split=0.2):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model = None

    def fit(self, X, y):
        self.model = self.build_fn()
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split, callbacks=[early_stopping], verbose=0)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def predict_proba(self, X):
        return self.model.predict(X) 


# Charger le modèle de deep learning
voting_classifier = joblib.load('voting_model.pkl')


#stacking

# Charger les modèles de base
lstm_model = load_model('lstm_model.h5')
gru_model = load_model('gru_model.h5')
cnn_model = load_model('cnn_model.h5')

# Charger le méta-modèle
meta_model = load('meta_model.joblib')




# Préparer une classe wrapper pour les modèles chargés
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def predict_proba(self, X):
        return self.model.predict(X)

# Wrappers pour les modèles chargés
lstm_classifier = KerasClassifierWrapper(lstm_model)
gru_classifier = KerasClassifierWrapper(gru_model)
cnn_classifier = KerasClassifierWrapper(cnn_model)



# Charger le tokenizer et le label encoder
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)




@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")





def predict_phrase(phrase, tokenizer, lstm_classifier, gru_classifier, cnn_classifier, meta_model):
    # Tokenisation et padding
    sequence = tokenizer.texts_to_sequences([phrase])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    # Obtenir les prédictions des modèles de base
    lstm_pred = lstm_classifier.predict_proba(padded_sequence)
    gru_pred = gru_classifier.predict_proba(padded_sequence)
    cnn_pred = cnn_classifier.predict_proba(padded_sequence)

    # Concatenate predictions to create meta model input
    meta_X = np.hstack((lstm_pred, gru_pred, cnn_pred))

    # Prédiction avec le méta-modèle
    final_pred = meta_model.predict(meta_X)

    return label_encoder.inverse_transform(final_pred)[0]

def gene_text(entre):
    logging.debug(f"Generating text for input: {entre}")
    if not entre.strip():
        raise ValueError("Le texte d'entrée ne peut pas être vide.")
    
    client = Groq(api_key="")
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "Tu es le meilleur en analyse de sentiments pour un supermarché qui vend des produits. Ton rôle est de donner le sentiment de chaque entrée de l'utilisateur. Tu dois absolument répondre par positive, negative ou neutral.",
            },
            {
                "role": "user",
                "content": entre
            }
        ],
        temperature=1,
        max_tokens=8,
        top_p=1,
        stream=True,
        stop=None,
    )

    generated_text = ""
    for chunk in completion:
        generated_text += chunk.choices[0].delta.content or ""

    texte = generated_text
    nouveau_texte = texte.replace("**", "").replace("* ", " ")
    logging.debug(f"Generated text: {nouveau_texte}")
    return nouveau_texte



@app.route("/predict", methods=["POST"])

def predict():
    try:
        content = request.json
        text_input = content['text']
        model_type = content.get('model', 'Voting')  # Par défaut, utilisez 'Voting' si non spécifié

        logging.debug(f"Model type requested: {model_type}")

        if model_type == 'LLM':
            logging.debug("Using LLM model for prediction.")
            prediction_label = gene_text(text_input)
        else:
            if model_type == 'Voting':
                logging.debug("Using Voting model for prediction.")
                model = voting_classifier
                sequences = tokenizer.texts_to_sequences([text_input])
                padded = pad_sequences(sequences, maxlen=100)
                prediction_indices = model.predict(padded)
                prediction_labels = label_encoder.inverse_transform(prediction_indices)
                prediction_label = prediction_labels[0]
            elif model_type == 'Stacking':
                logging.debug("Using Stacking model for prediction.")
                model = meta_model
                prediction_label = predict_phrase(text_input, tokenizer, lstm_classifier, gru_classifier, cnn_classifier, model)
            elif model_type == 'rf':
                logging.debug("Using Random Forest model for prediction.")
                model = joblib.load('random_forest.pkl')
                prediction_indices = model.predict([text_input])
                prediction_labels = label_encoder.inverse_transform(prediction_indices)
                prediction_label = prediction_labels[0]
            # Ajouter d'autres types de modèles ici

        logging.debug(f"Prediction result: {prediction_label}")
        return jsonify({"prediction": prediction_label})
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)})




@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        logging.debug("CSV file loaded successfully.")

        model_type = request.form.get('model', 'Voting')  # Par défaut, utilisez 'Voting' si non spécifié
        text_column = request.form.get('text_column')  # Nom de la colonne contenant le texte
        logging.debug(f"Model type requested: {model_type}")
        
        # Si le nom de la colonne n'est pas spécifié, deviner la colonne contenant le texte
        if not text_column:
            for col in df.columns:
                if df[col].dtype == object:  # Vérifier si la colonne contient des chaînes de caractères
                    text_column = col
                    break
        logging.debug(f"Text column determined: {text_column}")

        if not text_column or text_column not in df.columns:
            logging.error("Unable to determine the text column. Please specify it explicitly.")
            return jsonify({"error": "Unable to determine the text column. Please specify it explicitly."})

        if model_type == 'Voting':
            logging.debug("Using Voting model for prediction.")
            sequences = tokenizer.texts_to_sequences(df[text_column].values)
            padded = pad_sequences(sequences, maxlen=100)
            prediction_indices = voting_classifier.predict(padded)
            prediction_labels = label_encoder.inverse_transform(prediction_indices)

        elif model_type == 'Stacking':
            logging.debug("Using Stacking model for prediction.")
            predictions = []
            for text_input in df[text_column].values:
                prediction_label = predict_phrase(text_input, tokenizer, lstm_classifier, gru_classifier, cnn_classifier, meta_model)
                predictions.append(prediction_label)
            prediction_labels = np.array(predictions)

        elif model_type == 'LLM':
            logging.debug("Using LLM model for prediction.")
            prediction_labels = df['text'].apply(gene_text)

        elif model_type == 'rf':
            logging.debug("Using Random Forest model for prediction.")
            model = joblib.load('random_forest.pkl')
            prediction_indices = model.predict(df[text_column].values)
            prediction_labels = label_encoder.inverse_transform(prediction_indices)

        # Ajouter les prédictions au DataFrame
        df['Prediction'] = prediction_labels
        logging.debug("Predictions added to DataFrame.")

        # Sauvegarder le DataFrame dans un fichier CSV
        output_csv = io.StringIO()
        df.to_csv(output_csv, index=False)
        output_csv.seek(0)
        logging.debug("DataFrame saved to CSV.")

        # Générer le graphique circulaire
        sentiment_counts = df['Prediction'].value_counts()
        plt.figure(figsize=(6, 6))
        sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title('Distribution des sentiments')
        plt.tight_layout()

        # Sauvegarder le graphique dans un fichier image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode('utf-8')
        logging.debug("Pie chart generated and saved.")

        return jsonify({
            "csv_data": output_csv.getvalue(),
            "graph": "data:image/png;base64," + graph_url
        })
    except Exception as e:
        logging.error(f"Error during CSV upload and processing: {str(e)}")
        return jsonify({"error": str(e)})



if __name__ == "__main__":
    app.run(debug=True)
