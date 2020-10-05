"""Module to implement Perturbation Strategies for time series."""


class Perturbation:
    """Base Perturbation with abstract methods."""

    def __init__(self):
        pass

    def slicing(self):
        pass

    def convert_to_on_off_features(self):
        pass

    def perturbate(self):
        pass

    def convert_to_origin_format(self):
        pass

    def calculate_distance(self):
        pass
