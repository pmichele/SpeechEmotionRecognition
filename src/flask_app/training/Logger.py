import os
import torch


class LossLogger:
    """This class is responsible for logging the average loss periodically"""
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.log_file = os.path.join(self.model_dir,"log_loss.txt")
        text_file = open(self.log_file, "w")
        text_file.close()
        self.loss = 0
        self.count = 0
        self.epoch = 0

    def add(self, loss):
        """Add the contribution of an individual sample to the total loss"""
        self.loss += loss
        self.count += 1

    def log(self, net):
        """Log the average loss seen so far"""
        self.epoch += 1
        with open(self.log_file, "a") as text_file:
            if self.count == 0:
                self.count = 1
            text_file.write(str(self.loss/self.count))
            text_file.write('\n')
        torch.save({'epoch': self.epoch,
                  'state_dict': net.state_dict()},
                   os.path.join(self.model_dir,
                   'net_last.pth'))
        self.loss = 0
        self.count = 0


class AccuracyLogger:
    """This class is responsible for logging the accuracy periodically"""
    def __init__(self, model_dir, save_best=False):
        self.model_dir = model_dir
        self.log_file = os.path.join(self.model_dir, "log_accuracy.txt")
        text_file = open(self.log_file, "w")
        text_file.close()
        self.save_best = save_best
        self.correct = 0
        self.count = 0
        self.best_accuracy = 0

    def add(self, output, target):
        """Add the contribution of an individual sample to the overall accuracy"""
        prediction = torch.argmax(output, dim=1)
        self.correct += (target.eq(prediction.long())).sum()
        self.count += output.shape[0]

    def log(self, net):
        """Report the accuracy of the model over all samples seen so far. Additionally,
            save the given model if it has the best performance so far.
            Args:
                net: the model of which we are measuring performance.
        """
        accuracy = float(self.correct) / self.count
        best_label = '' # it's useful to log whether this model is the best so far
        if self.save_best and accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save({'state_dict': net.state_dict()},
                       os.path.join(self.model_dir, 'net_best_accuracy.pth'))
            best_label = 'best'
        with open(self.log_file, "a") as text_file:
            text_file.write('{}, {}\n'.format(accuracy, best_label))
        self.count = 0
        self.correct = 0