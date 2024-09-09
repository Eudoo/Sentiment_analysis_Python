# Sentiment Analysis project
  Sentiment analysis project using Tfidfvectorizer & Word2vec(gensim)

... The project is a first step for us in the field of NLP. The challenge was to train a model enable to predict differents sentiments according to the input (single word, sentence or text). We didn't do it exactly right now. 

We've tried differents approaches. First we use TfidfVectorizer with sklearn librairies but we notice some relevant bugs. So we dive into TextBlob, Word2vec and Bert pre-trained model for to capture the semantics between words in our inputs. We've used SVC and SGDC (Stochastic Gradient descent Classifer) to trainthe final models.

You can test our work in the following link :)) 👉🏾

## Testing in local
  1. Create a new Folder and open your Code editor.
  
  2. Create your virtual environment:
  
    _ Windows 
    
      ```
      python -m venv mon_env
      .\mon_env\Scripts\activate
      ```
      
    _ Linux 
    
      ```
      python3 -m venv mon_env
      source mon_env/bin/activate
      ```
      
  3. Clone the reposit :

    ``` 
    (mon_env) git clone https://github.com/rosasbehoundja/Sentiment_analysis_Python.git
    ```
    
  4. Move into the main directory : 
  
    ```
    (mon_env) cd PIL1_2324_2
    ```
    
  5. Install all the dependencies : 
  
    ```
    (mon_env) pip install -r requirements.txt*
    ```
    
  6. Then you can work with our notebooks or make changes.
  
  7. If you want to test the streamlit app in local, do : 
  
    ```
    (mon_env)  cd  '.\Without API\'
    (mon_env) streamlit run .\app_streamlit_without_api.py
    ```

Now you can enjoy 😉.

Please don't hesitate to make comment or do recommandation.

* [Rosas Behoundja](https://www.linkedin.com/in/rosas-behoundja-690513296?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
* [Jean-Eudes Codo](https://www.linkedin.com/in/eudes-codo-1b0b9a296?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
