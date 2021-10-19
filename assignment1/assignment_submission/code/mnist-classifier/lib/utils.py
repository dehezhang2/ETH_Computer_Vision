# Updating mean class for loss aggregation.
class UpdatingMean():
    def __init__(self):
        self.sum = 0
        self.n = 0

    def mean(self):
        return self.sum / self.n

    def add(self, loss):
        self.sum += loss
        self.n += 1