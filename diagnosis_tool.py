import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    "symptoms": ["headache, fever", "cough, sore throat", "nausea, vomiting"],
    "diagnosis": ["flu", "cold", "food poisoning"],
    "medicine": ["acetaminophen", "dextromethorphan", "ondansetron"]
}

df = pd.DataFrame(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["symptoms"], df["diagnosis"], test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit vectorizer to training data and transform both training and testing data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Define function to generate diagnosis and medicine recommendation
def diagnose_and_recommend(symptoms):
    symptoms_tfidf = vectorizer.transform([symptoms])
    diagnosis = clf.predict(symptoms_tfidf)[0]
    medicine = df.loc[df["diagnosis"] == diagnosis, "medicine"].iloc[0]
    return diagnosis, medicine

# Test the function
symptoms = "headache, fever"
diagnosis, medicine = diagnose_and_recommend(symptoms)
print(f"Diagnosis: {diagnosis}")
print(f"Medicine: {medicine}")
