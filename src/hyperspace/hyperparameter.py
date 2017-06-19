import numpy as np

NUMERTICAL_DTYPES = [int, float]

class Hyperparam(object):
    """Base class for hyperparameter objects."""
    def __init__(self, name):
        self.name = name


class NumericalHyperparam(Hyperparam):
    def __init__(self, name, minimum, maximum, scale='log', data_type='float'):
        if type(minimum) not in NUMERTICAL_DTYPES or type(maximum) not in NUMERTICAL_DTYPES:
            raise TypeError("minimum and maximum must be of numerical type.")
        if scale not in ['log', 'linear']:
            raise ValueError("scaling strategy must be one of \'log\' and \'linear\'.")
        self.min = minimum
        self.max = maximum
        self.scale = scale

        self.data_type = data_type

        self.n = None

        super(NumericalHyperparam, self).__init__(name)

    def adjust_min(self, new_min):
        self.min = new_min

    def adjust_max(self, new_max):
        self.max = new_max

    def adjust_range(self, new_min, new_max):
        self.adjust_min(new_min)
        self.adjust_max(new_max)

    def set_grid(self, n):
        self.n = n

    def grid(self):
        if self.n is None:
            raise ValueError("You need to set the grid scale value by calling .set_grid(n)")

        if self.scale == 'log':
            grid = np.logspace(self.min, self.max, self.n)
        elif self.scale == 'linear':
            grid = np.linspace(self.min, self.max, self.n)
        else:
            raise ValueError("Check that scale is one of log and linear")

        if self.data_type == 'int':
            int_grid = []
            for x in grid:
                int_grid.append(int(x))

            grid = int_grid

        return grid


class CategoricalHyperparam(Hyperparam):
    def __init__(self, name, options):
        if type(options) is not list:
            raise TypeError("categorical options must be given as a list.")
        self.options = options
        self.options2idx = {option: i for i, option in enumerate(self.options)}

        super(CategoricalHyperparam, self).__init__(name)

    def add_option(self, option):
        self.options2idx[option] = len(self.options)
        self.options.append(option)

    def remove_option(self, option):
        if option not in self.options:
            print("%s is already not an option" % option)
        del self.options[self.options2idx[option]]
        del self.options2idx[option]

    def grid(self):
        return self.options

