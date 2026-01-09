from collections import UserDict


class ParameterSet(UserDict):
    """
    A class that extends UserDict to represent a set of parameters.
    It allows for easy access and manipulation of parameters as if they were attributes.
    """

    def __init__(self, dict=None, **kwargs):
        # Initialize the data attribute directly to avoid recursion
        super().__init__(dict, **kwargs)
        self.modify_free_parameters = None

    def __getattr__(self, item):
        if item in self.data:
            return self.data[item]
        raise AttributeError(f"Parameter '{item}' not found in {self.__class__.__name__}.")

    def __setattr__(self, key, value):
        if key == 'data':  # Prevent recursion during initialization
            object.__setattr__(self, key, value)
        else:
            self.data[key] = value

    def __delattr__(self, item):
        if item in self.data:
            del self.data[item]
        else:
            raise AttributeError(f"Parameter '{item}' not found in {self.__class__.__name__}.")

    @property
    def cell_parameters(self):
        """
        Returns the cell parameters from the parameter set.
        """
        return self.__getattr__('cell_parameters')

    @property
    def free_parameters(self):
        """
        Returns the free parameters from the parameter set.
        """
        return self.__getattr__('free_parameters')

    def __getstate__(self):
        """Return the state (the data dictionary) to be pickled."""
        return self.data

    def __setstate__(self, state):
        """Restore the state from the pickled state."""
        # The 'state' is the dictionary returned by __getstate__.
        # Assigning to self.data is safe because our __setattr__ handles the 'data' key.
        self.data = state

    def __call__(self, **kwargs):
        """invokes parameter update on call."""
        import copy
        new_set = copy.deepcopy(self)
        for k, v in kwargs.items():
            if k in new_set:
                if isinstance(new_set[k], dict) and isinstance(v, dict):
                    new_set[k].update(v)
                else:
                    new_set[k] = v
            else:
                new_set[k] = v
        return new_set