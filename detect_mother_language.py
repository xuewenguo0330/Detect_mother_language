import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import nltk
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

def nlp_tfidf_classification(X,y_encoded):
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    X_tfidf = X_tfidf.toarray()

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_tfidf.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(11, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=60, validation_split=0.1)
    
    y_pred = model.predict(X_test).argmax(axis=1)
    return y_pred,y_test

def load_data():
    f = open("train.txt",'r')

    '''
        Les données sont enregistrés sur df avec deux colonnes 
        colonne 'label' => langage maternelle
        colonne 'doc' => phrases en anglais 
    '''
    label=[]
    phrase=[]
    lines = f.readlines()
    for l in lines:
        label.append(l[:5])
        phrase.append(l[5:])
    df = pd.DataFrame(list(zip(label, phrase)), columns =['label', 'doc'])
    Y=df["label"].tolist()
    le = LabelEncoder()
    le.fit(df.label.unique())
    return df, le.fit_transform(Y)

def main():
    df, y = load_data()
    y_pred,y_test = nlp_tfidf_classification(df["doc"],y)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
if __name__ == "__main__":
    main()