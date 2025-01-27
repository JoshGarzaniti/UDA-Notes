import regex as re
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
#import spacy #Having trouble with installing spacy in the console
 # python -m spacy download en_core_web_lg
import pandas as pd
import pyarrow
#Don't Run these in the console
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')


Last_Statements = pd.read_csv(r"C:\Unstructured Data Analytics\last_statements.csv")

print(Last_Statements.head(20))

nan_count = Last_Statements.isna().sum()
print(nan_count)
# We have 2 Na's in this that we have to deal with 
# so I can probably remove them and then move forward

Last_Statements = Last_Statements.dropna()

stemmer = PorterStemmer()
lematizer = nltk.WordNetLemmatizer()

Statements = Last_Statements['statements']

print(Statements)

len(Statements)
#Stemming example
Stemmed_Statements = [" ".join([stemmer.stem(word) 
                                for word in nltk.word_tokenize(statement)]) 
                                for statement in Statements]

#Lematization example
Lemmatized_Statements = [" ".join([lemmatizer.lemmatize(word) 
                                   for word in nltk.word_tokenize(statement)]) 
                                   for statement in Statements]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(Statements)

tfidf_matrix.toarray()

vocabulary = vectorizer.get_feature_names_out()

vocabulary

unique_vocabulary = list(vocabulary)

unique_vocabulary
