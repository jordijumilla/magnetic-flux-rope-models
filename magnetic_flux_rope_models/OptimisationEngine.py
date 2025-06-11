import numpy as np

class OptimisationParameter:
    def __init__(self,
                 name: str,
                 options: dict[str, str | float | tuple | list | np.ndarray]
                 ) -> None:
    
        # Every parameter must have a name, which has to match the model parameter name.
        if not isinstance(name, str) or name == "":
            raise ValueError("The optimisation parameter name must be a non-empty string.")
        
        self.name: str = name

        # Parse the mode.
        mode: str = options["mode"]
        if not isinstance(mode, str) or mode not in ["fixed", "optimised"]:
            raise ValueError(f"The optimisation mode name must be a string equal to 'fixed' or 'optimised', not {mode}.")
        self.mode: str = mode
        
        # Parse the type.
        type: str | None = options.get("type", None)
        if type is None:
            type = "float"
        elif not isinstance(type, str) or type not in ["float", "int"]:
            raise ValueError
        self.type = type

        if self.mode == "fixed":
            # The parameter is fixed, no optimisation is needed.
            self.fixed_value = options["value"]
            self.bounds = None
            self.initial_value = None
        elif self.mode == "optimised":
            # The parameter is to be optimised.
            self.fixed_value = None
            bounds = options.get("bounds", None)

            if bounds is not None:
                # Check that the bounds are valid.
                if len(bounds) != 2:
                    raise ValueError("The bounds must contain two values: the lower and upper bounds.")
                
                if not (bounds[0] < bounds[1]):
                    raise ValueError("The lower bound must be strictly smaller than the upper bound.")
            
            self.bounds = bounds

            # Set the initial value. If it is None, default to the mean value of the bounds.
            initial_value = options.get("initial_value", None)
            self.initial_value = initial_value if initial_value is not None else (bounds[0] + bounds[1]) / 2
        else:
            raise ValueError(f"OptimisationParameter cannot be of mode {self.mode}.")
    
    def __repr__(self) -> str:
        return f'{self.mode} optimisation parameter "{self.name}" of type "{self.type}"'


class OptimisationEngine:
    def __init__(self, optimisation_parameters: dict[str, dict[str, str | float]], model = None):
        self.optimisation_parameters: list[OptimisationParameter] = [OptimisationParameter(name=op_name, options=op_options) for op_name, op_options in optimisation_parameters.items()]

    