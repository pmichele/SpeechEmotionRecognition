import torch.nn as nn


class CNN(nn.Module):
    """This CNN model is designed for speech emotion classification. It expects
        a vector of concatenated features of size 193 as input. The architecture
        is fine-tuned for the emo-db dataset.
    """
    def __init__(self):
        super().__init__()
        n_classes = 7
        stride = 1
        kernel_size = 5
        padding = kernel_size // 2
        self.gpu = False                # whether this model is configured to run on gpu
        self.unit0 = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.down_pooling = nn.MaxPool1d(8)
        self.unit1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.dense = nn.Linear(3072, n_classes)

    def cuda(self):
        self.gpu = True
        return super().cuda()

    def forward(self, x):
        x1 = self.unit0(x)
        x1_pooled = self.down_pooling(x1)
        x2 = self.unit1(x1_pooled)
        x2_flattened = x2.view(x2.size(0), -1)
        prediction = self.dense(x2_flattened)
        return prediction
