import streamlit as st
import requests

# Conteneur pour le titre et le champ de texte
title_container = st.container()
info_container = st.container()
chat_container = st.container()
input_container = st.container()


# Affichage du titre dans le conteneur
with title_container:
    st.title('Analyseur de sentiment')

with info_container:
    st.write(''' ###### Entrez une phrase ou un paragraphe et on vous déduit le sentiment dominant ..!''')


def send_message():
    if st.session_state.new_message:
        # Récupérer le message saisi
        message = st.session_state.new_message

        # Envoie de la requete
        req = requests.post(url='http://localhost:8080/predict', json={'message': message})
        predicted_label = req.json()['prediction']

        # Générer une réponse
        label = {0: 'Triste', 1: 'Joie', 3: 'colère', 4: 'peur'}
        sentiment = label[predicted_label]

        # Afficher le message et la réponse dans le conteneur de discussion
        with chat_container:
            st.write(f'__Vous__ : {message}')
            st.write(f'''__Réponse__ : Le sentiment de cette phrase est __{sentiment}__''')
        # Réinitialiser le champ de saisie
        st.session_state.new_message = ""


# Champ de saisie pour un nouveau message dans le conteneur de saisie
with input_container:
    st.text_input("Entrez un message:", key="new_message", on_change=send_message)

