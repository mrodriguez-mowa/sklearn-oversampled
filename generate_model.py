import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn.model_selection import train_test_split

df = pd.read_csv("out_parsed.csv")

df = df.drop_duplicates()

grouped = df.groupby('label').size().reset_index(name='count')
sorted_df = grouped.sort_values('count', ascending=False)

max_count = grouped['count'].max()

# Oversample the data
oversampled_data = pd.DataFrame()
for _, row in grouped.iterrows():
    label = row['label']
    count = row['count']
    oversampled_rows = df[df['label'] == label].sample(n=max_count, replace=True, random_state=42)
    oversampled_data = pd.concat([oversampled_data, oversampled_rows])

# Reset the index of the oversampled data
oversampled_data = oversampled_data.reset_index(drop=True)

grouped_os = oversampled_data.groupby('label').size().reset_index(name='count')
sorted_os = grouped_os.sort_values('count', ascending=False)

# Adding numeric values to each label
labels = np.unique(df['label'])

def assign_label_id(row, label_ids):
    label = row['label']
    return label_ids[label]

def assign_label_ids(df, label_column):
    unique_labels = df[label_column].unique()
    label_ids = {label: i+1 for i, label in enumerate(unique_labels)}
    df['label_id'] = df.apply(lambda row: assign_label_id(row, label_ids), axis=1)
    return df

oversampled_df = assign_label_ids(oversampled_data, 'label')

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(oversampled_df['message'])
y = oversampled_df['label']

X_train_oversampled, X_val_oversampled, y_train_oversampled, y_val_oversampled = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_oversampled, y_train_oversampled)

# Evaluate the model on the oversampled validation/test data
accuracy_oversampled = model.score(X_val_oversampled, y_val_oversampled)
print("Oversampled Data - Accuracy: {:.2f}".format(accuracy_oversampled))

# Save the model
dump(model, 'model.joblib')
dump(vectorizer, 'vectorizer.joblib')