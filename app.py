from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_session import Session
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load the pre-trained model
model = load_model('baseline_cnn_mitbih.keras')

# Define the label mapping
label_map = {
    0: 'Non-ecotic beat (Normal beat)',
    1: 'Supraventricular ectopic beats',
    2: 'Ventricular ectopic beats',
    3: 'Fusion Beats',
    4: 'Unknown beats'
}

def get_label(prediction):
    predicted_class = np.argmax(prediction)
    return label_map[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                df = pd.read_csv(file)
                session['df'] = df.to_json()
                max_index = df.shape[0] - 1
                return render_template('index.html', max_index=max_index)
            except Exception as e:
                flash(f'Failed to process file: {str(e)}', 'error')
    return render_template('index.html', max_index=None)

@app.route('/plot', methods=['POST'])
def plot_signal():
    signal_index = int(request.form['signal_index'])
    df_json = session.get('df')

    if df_json is None:
        flash('Session expired or data not found, please upload the file again.', 'error')
        return redirect(url_for('upload_file'))

    try:
        df = pd.read_json(df_json)
        signal = df.iloc[signal_index].values
        trace = go.Scatter(x=list(range(len(signal))), y=signal, mode='lines', name=f'Signal {signal_index}')
        layout = go.Layout(
            title=f'Signal {signal_index}',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Voltage'),
            template='plotly_white'
        )
        fig = go.Figure(data=[trace], layout=layout)
        plot_div = fig.to_html(full_html=False)
        
        prediction = model.predict(signal.reshape(1, signal.shape[-1], 1))
        predicted_label = get_label(prediction)
        
        return render_template('plot.html', plot_div=plot_div, predicted_label=predicted_label)
    except Exception as e:
        flash(f'An error occurred during processing: {str(e)}', 'error')
        return redirect(url_for('upload_file'))

@app.route('/plot_all', methods=['POST'])
def plot_all_signals():
    df_json = session.get('df')

    if df_json is None:
        flash('Session expired or data not found, please upload the file again.', 'error')
        return redirect(url_for('upload_file'))

    try:
        df = pd.read_json(df_json)
        traces = []
        predictions = []

        for index, signal in df.iterrows():
            signal_values = signal.values
            trace = go.Scatter(x=list(range(len(signal_values))), y=signal_values, mode='lines', name=f'Signal {index}')
            traces.append(trace)
            prediction = model.predict(signal_values.reshape(1, signal_values.shape[-1], 1))
            predictions.append((index, get_label(prediction)))  # Store index along with the label
        
        layout = go.Layout(
            title='All Signals',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Voltage'),
            template='plotly_white'
        )
        fig = go.Figure(data=traces, layout=layout)
        plot_div = fig.to_html(full_html=False)
        
        return render_template('plot_all.html', plot_div=plot_div, predictions=predictions)
    except Exception as e:
        flash(f'An error occurred during processing: {str(e)}', 'error')
        return redirect(url_for('upload_file'))

@app.route('/select_index', methods=['GET'])
def select_index():
    df_json = session.get('df')
    if df_json is None:
        flash('Session expired or data not found, please upload the file again.', 'error')
        return redirect(url_for('upload_file'))
    df = pd.read_json(df_json)
    max_index = df.shape[0] - 1
    return render_template('index.html', max_index=max_index)

if __name__ == '__main__':
    app.run(debug=True)
