import numpy as np


class RandomNoise:
    def __init__(self, random_seed: int = 0) -> None:
        # Initialise the random number generator using the random seed provided.
        self.random_number_generator = np.random.default_rng(seed=random_seed)

        # Store the random seed in case it may need to be reused or accessed later on.
        # This seed is important to obtain repeatable results.
        self._random_seed: int = random_seed

    def generate_noise(self, num_samples: int) -> np.ndarray:
        raise NotImplementedError


class UniformNoise(RandomNoise):
    def __init__(self, epsilon: float) -> None:
        super().__init__()
        self.epsilon: float = epsilon

    def generate_noise(self, num_samples: int) -> np.ndarray:
        """Generate num_samples of uniformly distributed noise."""
        return self.epsilon * (2 * self.random_number_generator.random(num_samples) - 1)


class GaussianNoise(RandomNoise):
    """Class that allows to generate Gaussian distributed samples of mean = mu and standard deviation = sigma."""

    def __init__(self, mu: float = 0, sigma: float = 0.05) -> None:
        super().__init__()
        self.mu: float = mu
        self.sigma: float = sigma

    def generate_noise(self, num_samples: int) -> np.ndarray:
        """Return an array of num_samples random samples of a Gaussian probability distribution."""
        return self.random_number_generator.normal(self.mu, self.sigma, num_samples)
