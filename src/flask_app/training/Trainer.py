from torch.autograd import Variable
import sys
import time


class Trainer:
    """This class is responsible for training a model."""
    def __init__(self, net, data_loader, optimizer, loss_function,
                 tester, logger, test_every_n_steps, lr_scheduler=None):
        self.net = net
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.crit = loss_function
        self.logger = logger
        self.data_stream = iter(self.data_loader)
        self.epoch = 0
        self.prev_iter = 0
        self.test_every_n_steps = test_every_n_steps
        self.tester = tester
        self.lr_scheduler = lr_scheduler
        if net.gpu:
            self.crit = self.crit.cuda()

    def train(self, num_iter):
        """Train the underlying model for a given number of steps. This function
            has no memory of previous calls.
        """
        self.net.train()
        tot_iter=0
        t0=time.time()
        while tot_iter < num_iter:
            try:
                sample, label = next(self.data_stream)
                label = label.long()
                self.optimizer.zero_grad()
                if self.net.gpu:
                    sample = sample.cuda()
                    label = label.cuda()
                sample = Variable(sample)
                prediction = self.net.forward(sample)
                loss = self.crit(prediction, label)
                loss.backward()
                self.optimizer.step()
                self.logger.add(loss.data.cpu().numpy())
                tot_iter += 1
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                t1=time.time()
                if t1 - t0 > 3:
                    time_per_iter = (t1 - t0) / (tot_iter - self.prev_iter)
                    sys.stdout.write('\rIter: %8d\tEpoch: %6d\tTime/iter: %6f' %
                                     (tot_iter, self.epoch, time_per_iter))
                    t0 = t1
                    self.prev_iter=tot_iter
                if self.test_every_n_steps and tot_iter % self.test_every_n_steps == 0:
                    self.tester.test()
                    self.logger.log(self.net)
            except StopIteration:
                self.epoch+=1
                self.data_stream = iter(self.data_loader)