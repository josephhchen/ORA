from sklearn.feature_extraction.text import CountVectorizer

text = ["I love python"]
vectorizer = CountVectorizer()
vectorizer.fit(text)
vector = vectorizer.transform(text).toarray()

print(vector)