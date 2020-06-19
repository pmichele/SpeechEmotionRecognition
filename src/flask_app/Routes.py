"""Module for handling user requests. It supports two endpoints, namely 'train' and 'predict'"""
from src.flask_app import app

import os
import json

from src import load_config
from src.flask_app.meta_architectures.CNN import CNN
from src.flask_app.training.ModelThread import ModelThread
from src.flask_app.Predictor import Predictor

from flask import request, jsonify, redirect

demo = True # In demo mode the trained model is bypassed by a pre-trained model during prediction

'''Load Hyper-parameters for each model'''
cnn_config = load_config("/src/flask_app/configs/cnn_config.json")


'''Threads for models training and testing'''
cnn_thread = ModelThread(cnn_config, CNN)
cnn_predictor = Predictor(cnn_config, CNN, demo=demo)

train_set_len = len(cnn_thread.train_loader) * cnn_config.batch_size


def model_progress():
    """Return the progress of the neural network"""
    loss, accuracy, best_accuracy, num_iters, epochs = 0, 0, 0, 0, 0
    loss_file = os.path.join(cnn_config.model_dir, "log_loss.txt")
    acc_file = os.path.join(cnn_config.model_dir, "log_accuracy.txt")
    if os.path.exists(loss_file):
        with open(os.path.join(cnn_config.model_dir, "log_loss.txt")) as f:
            for line in f:
                loss = float(line.strip())
    if os.path.exists(acc_file):
        with open(acc_file) as f:
            best_accuracy = 0
            for line in f:
                num_iters += 1
                accuracy = float(line.split(",")[0])
                best_accuracy = max(best_accuracy, accuracy)
    num_iters *= cnn_config.test_every_n_steps
    epochs = num_iters // train_set_len
    return jsonify({
        "iter": num_iters,
        "loss": loss,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "epochs": epochs
    })


@app.route("/train")
@app.route("/train/cnn")
def run_cnn():
    if cnn_thread.ident is not None:
        return model_progress()
    # Start the training of the neural network
    cnn_thread.start()
    with open(os.path.join(cnn_config.model_dir, "cnn_config.json"), 'w') as copy:
        copy.write(json.dumps(cnn_config))
    return "Starting training for model {}, please refresh to update".format(cnn_config.model_dir)


@app.route("/")
def home():
    return redirect("/train")


@app.route("/predict")
@app.route("/predict/cnn")
def predict_cnn():
    if not demo:
        if cnn_thread.is_alive():
            # As we don't need predictions during training, do nothing
            return "This model is currently training. Please retry after completion."
        if not os.path.exists(cnn_predictor.model_dir):
            return "Please train the model before running predictions."
    print("Running prediction on model ", cnn_config.model_dir)
    sample = request.args.get("sample")
    if sample is None:
        return jsonify(list(cnn_predictor.filename_to_index_map.keys()))
    if sample not in cnn_predictor:
        return "Sample not found."
    _, pLabel, _, tLabel, confidence = cnn_predictor.predict(sample)
    return jsonify({
        'prediction': pLabel,
        'truth': tLabel,
        'confidence': confidence
    })
