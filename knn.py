import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


data=[]
with open('D:/dataset/nn/splice.data', 'r') as f:
    for line in f:
        parts=line.strip().split(',')
        Label=parts[0]
        Sequence=parts[2].replace(" ","")
        data.append((Label,Sequence))

df=pd.DataFrame(data, columns=['Label', 'Sequence'])

vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1))
X = vectorizer.fit_transform(df['Sequence'])

le = LabelEncoder()
y = le.fit_transform(df['Label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("KNN Classification Report:")
print(classification_report(y_test, knn.predict(X_test), target_names=le.classes_))


nb = MultinomialNB()
nb.fit(X_train, y_train)
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, nb.predict(X_test), target_names=le.classes_))
