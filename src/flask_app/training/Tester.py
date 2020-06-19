import torch
import sys
import time


class Tester:
    """This class is responsible for testing a model"""
    def __init__(self, net, test_loader, logger):
        self.logger = logger
        self.data_loader = test_loader
        self.data_stream = iter(test_loader)
        self.net = net

    def test(self):
        """Test the model on the underlying test set"""
        self.net.eval()
        with torch.no_grad():
            tot_iter = 0

            t0 = time.time()
            while True:
                try:
                    sample, label = next(self.data_stream)
                    if self.net.gpu:
                        sample = sample.cuda()
                        label = label.cuda()
                    prediction = self.net.forward(sample)
                    self.logger.add(prediction, label)
                    tot_iter += 1
                    t1 = time.time()
                    if t1 - t0 > 3:
                        sys.stdout.write('\rTest iter: %8d' % tot_iter)
                        t0 = t1
                except StopIteration:
                    self.logger.log(self.net)
                    break
        self.data_stream = iter(self.data_loader)
        self.net.train()

