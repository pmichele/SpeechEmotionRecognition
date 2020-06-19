import torch
import os
from src.flask_app.MultiFeaturesDataset import MultiFeaturesDataset
import pandas as pd


class Predictor:
    """This class is responsible for running model predictions on test samples."""
    def __init__(self, config, model_builder, demo=False):
        """Build the model and the input pipeline"""
        model = model_builder()
        self.model_dir = "/models/best_cnn/"  # use pre-computed weights for the demo
        if not demo:
            self.model_dir = config.model_dir
        self.model = model
        self.dataset = MultiFeaturesDataset(config.query_set_dir, config.label_map_path, "predict")
        self.config = config
        self.filename_to_index_map = {}  # reverse mapping from filename to dataset index
        # used for fast query lookup
        for idx, file_name in enumerate(self.dataset.names):
            _, file_name = os.path.split(file_name)
            self.filename_to_index_map[file_name] = idx
        self.memo = [None] * len(self.dataset)  # Cache computed predictions
        self.initialized_weights = False

    def _update(self):
        """Load the model weights, but only for the first query."""
        if self.initialized_weights:
            return
        self.initialized_weights = True
        device = "cpu"
        if self.config.gpu:
            self.model.cuda()
            device = "cuda"
        saved_net = torch.load(os.path.join(self.model_dir, "net_best_accuracy.pth"),
                               map_location=torch.device(device))
        self.model.load_state_dict(saved_net['state_dict'])
        self.model.eval()

    def _predict(self, i):
        """Compute prediction for an item in the dataset. Results are cached so that
            computation is amortized.
        """
        if self.memo[i] is not None:
            return self.memo[i]
        self._update()
        x, y = self.dataset[i]
        x = x.unsqueeze(0)
        out = torch.nn.functional.softmax(self.model.forward(x), dim=1)
        p = torch.argmax(out, dim=1)[0].tolist()
        confidence = out[0, p].tolist()
        prediction_label = self.dataset.class_index_to_label_map[p]
        truth_label = self.dataset.class_index_to_label_map[y]
        self.memo[i] = (p, prediction_label, y, truth_label, confidence)
        return self.memo[i]

    def predict(self, utterance):
        """Compute prediction for a given test sample"""
        return self._predict(self.filename_to_index_map[utterance])

    def predict_all(self):
        """Compute predictions for all samples in the dataset and return the result
            in a pandas dataframe
        """
        ans = pd.DataFrame(columns=['Prediction', 'Prediction Label',
                                    'Truth', 'Truth Label', 'Confidence'])
        for i in range(len(self.dataset)):
            ans.loc[i] = self._predict(i)
        ans['Filename'] = [os.path.split(filename)[1] for filename in self.dataset.names]
        ans.set_index('Filename', inplace=True)
        return ans

    def __contains__(self, key):
        return key in self.filename_to_index_map

    def reload(self):
        self.initialized_weights = False
