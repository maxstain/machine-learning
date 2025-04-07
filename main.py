import os

try:
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from flask import Flask, request, render_template, jsonify
    import joblib
except ImportError as e:
    print(f"Some modules are missing, installing them now...")
    os.system('pip install -r requirements.txt')


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_clean_data(self):
        df = pd.read_csv(self.file_path)
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
        return df


class ModelTrainer:
    def __init__(self, df):
        self.df = df

    def perform_clustering(self):
        kmeans = KMeans(n_clusters=3)
        self.df['Cluster'] = kmeans.fit_predict(self.df.drop('Claim', axis=1))
        print(self.df['Cluster'].value_counts())

    def train_model(self):
        X = self.df.drop('Claim', axis=1)
        y = self.df['Claim']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        joblib.dump(model, 'model.pkl')


def create_app():
    app = Flask(__name__)
    model = joblib.load('model.pkl')

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        data_df = pd.DataFrame([data])
        prediction = model.predict(data_df)
        return jsonify({'prediction': int(prediction[0])})

    @app.route('/')
    def home():
        return render_template('index.html')

    return app


if __name__ == '__main__':
    data_processor = DataProcessor('PROJET6/dataset.csv')
    df = data_processor.load_and_clean_data()

    model_trainer = ModelTrainer(df)
    model_trainer.perform_clustering()
    model_trainer.train_model()

    app = create_app()
    app.run(debug=True)
