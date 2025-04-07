# Imports
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template, jsonify
import joblib

# 1. Data cleaning
df = pd.read_csv('PROJET6/dataset.csv')
print(df.info())
df = df.dropna()
df = pd.get_dummies(df, columns=[
    'Agency',
    'Agency Type',
    'Distribution Channel',
    'Product Name',
    'Destination',
    'Gender'
], drop_first=True)
print(df.head())

# 2. Clustering
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(df.drop('Claim', axis=1))
print(df['Cluster'].value_counts())

# 3. Prediction
X = df.drop('Claim', axis=1)
y = df['Claim']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the model as a pickle file to be read by Flask
joblib.dump(model, 'model.pkl')

# 4. Flask API
app = Flask(__name__)

model = joblib.load('model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        prediction = model.predict([data['features']])
        return jsonify({'prediction': prediction[0]})
    else:
        return jsonify({'error': 'Unsupported Media Type'}), 415


# The home route to index.html
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
