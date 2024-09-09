import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import joblib
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Pour les données du WordNet Lemmatizer



# Conteneur pour le titre et le champ de texte
title_container = st.container()
info_container = st.container()
chat_container = st.container()
input_container = st.container()



def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in ['feel', 'feeling']]
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [str(lemmatizer.lemmatize(token)) for token in filtered_tokens]
    return lemmatized_tokens

def vectorize_text(text, model):
    word_vectors = [model.wv[word] for word in text if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
    


def charger_modele():
    # Charger le modèle du vecteur word2vec avec joblib
    modele = joblib.load('Models/modele_word2vec.joblib')
    return modele


def charger_svc():
    # Charger le modèle svc avec joblib
    svc = joblib.load('Models/svc_model.joblib')
    #svc = joblib.load('Models/sgdc_rl_model.joblib')
    return svc

    
# Affichage du titre dans le conteneur
with title_container:
    st.title('Analyseur de sentiment')

with info_container:
    st.write(''' ###### Entrez une phrase ou un paragraphe et on vous déduit le sentiment dominant ..!''')


def send_message():
    if st.session_state.new_message:
        # Récupérer le message saisi
        message = st.session_state.new_message
        
        preprocessed_sentence = preprocess_text(message)

        # Étape 2 : Vectoriser la phrase
        sentence_vector = vectorize_text(preprocessed_sentence, charger_modele())

        # Étape 3 : Prédire le label avec les modèles entraînés
        predicted_label_rf = charger_svc().predict([sentence_vector])[0]

        # Générer une réponse
        label = {0: 'Triste', 1: 'Joie', 3: 'Colère', 4: 'Peur'}
        sentiment = label[predicted_label_rf]

        # Générer une réponse (ici, on génère une réponse simple)
        # prediction = '...'
        response = f"Le sentiment de cette phrase est __{sentiment}__"
        # Afficher le message et la réponse dans le conteneur de discussion
        with chat_container:
            st.write(f''' __Vous__ : {message}''')
            st.write(f"__Réponse__ : {response}")
        # Réinitialiser le champ de saisie
        st.session_state.new_message = ""

# Champ de saisie pour un nouveau message dans le conteneur de saisie
with input_container:
    st.text_input("Entrez un message:", key="new_message", on_change=send_message)

