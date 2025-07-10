import math


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(
        self,
        cyclical_annealing,
        total_steps,
        shape,
        annealing_scale=1,
        baseline=0.0,
        cyclical=True,
        disable=False,
    ):
        """
        Args:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        if cyclical_annealing:
            print("Using cyclical annealing.")
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        self.annealing_scale = annealing_scale

        if disable:
            self.shape = "none"
            self.baseline = 0.0

    def get_beta(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = y * self.annealing_scale
        y += self.baseline

        return y

    def step(self):

        if self.current_step < self.total_steps:
            self.current_step += 1

        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0

        if ~self.cyclical and self.current_step >= self.total_steps:
            self.shape = "none"
            self.baseline = 0.0

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
