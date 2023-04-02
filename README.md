# Projet Statistique -- GUO Xuewen, NDAO Ndieme
## Abstrat 
Nous avons choisi d'implémenter ce projet en utilisant le langage de programmation Python.
Le but du projet est de détecter la langue maternelle à partir de phrases écrites en anglais.

## Prerequisites
|  bibliotheque utilisé    | version  |
|  ----  | ----  |
| tensorflow  | 2.12.0 |
| keras  | 2.12.0 |
| nltk  | 3.8.1|
| scikit-learn  |  1.2.1|
|Python|3.11.1|

## Fichiers
Modele finale: 
- "detect_mother_language.py": Modele finale

Modeles réalisés pour tester:
- "prediction_crossValidation.ipynb": Modeles SVM et Logistic Regression avec Cross Validation par KFold
- "prediction_RNN.ipynb": Modele RNN 
- "prediction_model_SVM_LR" : Modeles SVM et Logistic Regression avec différents façon de vectoriser (TF-IDF et CountVectorizer)



## Run the application
Sur terminal, `python3 detect_mother_language.py`





