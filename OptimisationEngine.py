import numpy as np

class OptimisationParameter:
    def __init__(self,
                 name: str,
                 fixed_value: float | None = None, bounds: tuple[float | None, float | None] | np.ndarray | list[float] | None = None,
                 initial_value: float | None = None) -> None:
        # Every parameter must have a name, which has to match the model parameter name.
        if not isinstance(name, str) or name == "":
            raise ValueError("The optimisation parameter name must be a non-empty string.")
        self.name: str = name

        if fixed_value is None and bounds is None and initial_value is None:
            # Free optimisation parameter.
            pass
        elif fixed_value is not None and bounds is None and initial_value is None:
            # The parameter is fixed, no optimisation is needed.
            self.fixed_value = fixed_value
            self.bounds = None
            self.initial_value = None
        elif fixed_value is None and bounds is not None:
            # The parameter is to be optimised.
            self.fixed_value = None

            # Check that the bounds are valid.
            if len(bounds) != 2:
                raise ValueError("The bounds must contain two values: the lower and upper bounds.")
            
            if not (bounds[0] < bounds[1]):
                raise ValueError("The lower bound must be strictly smaller than the upper bound.")
            
            self.bounds = bounds

            # Set the initial value. If it is None, default to the mean value of the bounds.
            self.initial_value = initial_value if initial_value is not None else (bounds[0] + bounds[1]) / 2
        else:
            raise ValueError("OptimisationParameter cannot have all 'fixed_value', 'bounds' and 'initial_value' equal to None.")


class OptimisationEngine:
    def __init__(self, optimisation_parameters: list[OptimisationParameter]):
        self.optimisation_parameters: list[OptimisationParameter] = optimisation_parameters
