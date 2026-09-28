
class Loss_Metric_Manager():
    def __init__(self, loss_names=None, metric_names=None):
        if type(loss_names) != type(None):
            self.losses = {l: [] for l in loss_names}
        else:
            self.losses = {}
        if type(metric_names) != type(None):
            self.metrics = {m: [] for m in metric_names}
        else:
            self.metrics = {}

        self.other = {}
        
    def make_new_loss(self, loss: str):
        self.losses[loss] = []
        
    def make_new_metric(self, metric: str):
        self.metrics[metric] = []
        
    def make_new_losses(self, losses: list):
        for l in losses:
            self.losses[l] = []
            
    def make_new_metrics(self, metrics: list):
        for m in metrics:
            self.metrics[m] = []
        
    def add_losses(self, losses: dict):
        for l in losses.keys():
            if l not in self.losses:
                self.make_new_loss(l)
            self.losses[l].append(losses[l])
        
    def add_metrics(self, metrics: dict):
        for m in metrics.keys():
            if m not in self.metrics:
                self.make_new_metric(m)
            self.metrics[m].append(metrics[m])

    def make_new_other(self, other: str):
        self.other[other] = []

    def add_other(self, other: dict):
        for m in other.keys():
            if m not in self.other:
                self.make_new_other(m)
            self.other[m].append(other[m])

