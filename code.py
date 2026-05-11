import pandas as pd
df = pd.read_csv('all-data.csv', encoding='latin-1', header=None)
df.columns = ['sentiment', 'text']
print(df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['sentiment'], 
    test_size=0.2, 
    random_state=42
)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

from sklearn.metrics import classification_report, accuracy_score
predictions = model.predict(X_test_tfidf)
print(f"Akurasi: {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))