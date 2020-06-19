from src.flask_app.MultiFeaturesDataset import MultiFeaturesDataset
from src.flask_app.training.Logger import LossLogger, AccuracyLogger
from src.flask_app.training.Trainer import Trainer
from src.flask_app.training.Tester import Tester
import torch
import torch.optim as optim
import os
from threading import Thread


def lr_lambda(e):
    """Learning rate decay function"""
    return 1/(1+e*1e-6)


class ModelThread(Thread):
    """This class is responsible for building, training and testing a model."""
    def __init__(self, config, model_builder):
        """Build the input pipeline"""
        Thread.__init__(self)
        self.config = config
        self.model_builder = model_builder
        self.train_loader = torch.utils.data.DataLoader(
            MultiFeaturesDataset(config.train_set_dir, config.label_map_path, "train"),
            batch_size=config.batch_size, shuffle=True,
            num_workers=4, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(
            MultiFeaturesDataset(config.test_set_dir, config.label_map_path, "test"),
            batch_size=config.batch_size, shuffle=False,
            num_workers=2, drop_last=False)
        if config.prev_model_dir is not None:
            device = "cpu"
            if config.gpu:
                device = "cuda"
            self.saved_net = torch.load(os.path.join(config.prev_model_dir, "net_last.pth"),
                                        map_location=torch.device(device))
        self.loss = torch.nn.CrossEntropyLoss()
        os.makedirs(self.config.model_dir)

    def run(self):
        """Build the model and perform its training and testing"""
        print("Starting training for model {}".format(self.config.model_dir))
        model = self.model_builder()
        if self.config.gpu:
            model.cuda()
        if self.config.prev_model_dir is not None:
            model.load_state_dict(self.saved_net['state_dict'])
        optimizer = optim.RMSprop(model.parameters(), lr=self.config.lr, weight_decay=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        testing_logger = AccuracyLogger(self.config.model_dir, save_best=True)
        training_logger = LossLogger(self.config.model_dir)
        tester = Tester(model, self.test_loader, testing_logger)
        trainer = Trainer(model, self.train_loader, optimizer, self.loss, tester,
                          training_logger, test_every_n_steps=self.config.test_every_n_steps,
                          lr_scheduler=lr_scheduler)
        trainer.train(self.config.num_iterations)

