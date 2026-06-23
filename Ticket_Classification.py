# CUSTOMER SUPPORT TICKET CLASSIFICATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# LOAD DATASET

df = pd.read_csv(r"C:/Users/bhara/OneDrive/Desktop/akhil/customer_support_tickets.csv")

# print("Dataset Shape:", df.shape)
# print(df.head())

# DATA PREPROCESSING

# Combine subject and description
df["text"] = (
    df["Ticket Subject"].fillna("") +
    " " +
    df["Ticket Description"].fillna("")
)

# Remove missing labels
df = df.dropna(subset=["Ticket Type", "Ticket Priority"])

# TICKET TYPE CLASSIFICATION

X = df["text"]
y_type = df["Ticket Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_type,
    test_size=0.2,
    random_state=42,
    stratify=y_type
)

ticket_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

ticket_model.fit(X_train, y_train)

type_pred = ticket_model.predict(X_test)

print("\n==============================")
print("TICKET TYPE CLASSIFICATION")
print("==============================")

print("Accuracy:",
      round(accuracy_score(y_test, type_pred), 4))

print(classification_report(y_test, type_pred))

# PRIORITY PREDICTION MODEL

y_priority = df["Ticket Priority"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X,
    y_priority,
    test_size=0.2,
    random_state=42,
    stratify=y_priority
)

priority_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

priority_model.fit(X_train2, y_train2)

priority_pred = priority_model.predict(X_test2)

print("\n==============================")
print("PRIORITY PREDICTION")
print("==============================")

print("Accuracy:",
      round(accuracy_score(y_test2, priority_pred), 4))

print(classification_report(y_test2, priority_pred))

# CONFUSION MATRIX - TICKET TYPE

cm = confusion_matrix(y_test, type_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=np.unique(y_test),
    yticklabels=np.unique(y_test)
)

plt.title("Ticket Type Classification Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# CONFUSION MATRIX - PRIORITY

cm2 = confusion_matrix(y_test2, priority_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm2,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=np.unique(y_test2),
    yticklabels=np.unique(y_test2)
)

plt.title("Priority Prediction Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# BUSINESS DASHBOARD VISUALIZATION

predicted_types = pd.Series(type_pred)

plt.figure(figsize=(8,5))
predicted_types.value_counts().plot(
    kind="bar"
)

plt.title("Predicted Ticket Categories")
plt.xlabel("Category")
plt.ylabel("Number of Tickets")
plt.xticks(rotation=30)
plt.show()

# ------------------------------------------

predicted_priority = pd.Series(priority_pred)

plt.figure(figsize=(8,5))
predicted_priority.value_counts().plot(
    kind="bar"
)

plt.title("Predicted Priority Distribution")
plt.xlabel("Priority Level")
plt.ylabel("Number of Tickets")
plt.xticks(rotation=0)
plt.show()

# NEW TICKET TESTING

sample_ticket = """
I was charged twice for my subscription.
Please refund the extra payment immediately.
"""

predicted_category = ticket_model.predict([sample_ticket])[0]
predicted_priority = priority_model.predict([sample_ticket])[0]

print("\nNEW TICKET ANALYSIS")
print("----------------------")
print("Ticket:", sample_ticket)
print("Category:", predicted_category)
print("Priority:", predicted_priority)