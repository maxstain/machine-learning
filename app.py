# app.py
import os
from flask import Flask, request, render_template, send_file, redirect, url_for, flash
import pandas as pd
from werkzeug.utils import secure_filename
from src.xgb_claim_predictor import predict_claim

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PREDICTIONS_FOLDER'] = 'predictions'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTIONS_FOLDER'], exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run prediction
            df = pd.read_csv(filepath)
            preds, probs = predict_claim(df)
            df['Predicted Claim'] = preds
            df['Claim Probability'] = probs

            result_path = os.path.join(app.config['PREDICTIONS_FOLDER'], f'predictions_{filename}')
            df.to_csv(result_path, index=False)

            flash('Prediction completed successfully!')
            return redirect(url_for('download_file', filename=f'predictions_{filename}'))

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['PREDICTIONS_FOLDER'], filename), as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
