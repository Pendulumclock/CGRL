import json
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import rankdata
from scipy.stats import norm
from scipy.interpolate import make_interp_spline

class GaussianCurriculumSampler:
    def __init__(self, difficulty_list, total_steps, mu_0=0.1, sigma=0.05):
        """
        difficulty_list: list or np.array, each sample's difficulty score d_i ∈ [0,1]
        total_steps: int, maximum training steps
        mu_0: float, initial mean of the sampling distribution
        sigma: float, standard deviation of the Gaussian distribution
        """
        self.difficulty = np.array(difficulty_list)
        self.total_steps = total_steps
        self.mu_0 = mu_0
        self.sigma = sigma
        self.N = len(difficulty_list)

    def get_mu(self, step):
        """
        Easy samples are favored early, harder samples later.
        μ(t) smoothly transitions from mu_0 to 1 as training progresses.
        """
        t = min(step, self.total_steps)
        return self.mu_0 + (1 - self.mu_0) * (t / self.total_steps)

    def compute_probabilities(self, step):
        mu = self.get_mu(step)
        # Compute the Gaussian sampling probability for each sample
        probs = np.exp(- ((self.difficulty - mu) ** 2) / (2 * self.sigma ** 2))
        probs = probs / probs.sum()  # Normalize
        return probs

    def sample_batch(self, step, batch_size):
        probs = self.compute_probabilities(step)
        indices = np.random.choice(self.N, size=batch_size, replace=False, p=probs)
        return indices.tolist()


def main():
    citynavDataPath = "data/data_example.json"
    with open(citynavDataPath, 'r') as f:
        citynavData = json.load(f)

    difficulties = []
    for data in citynavData:
        difficulties.append(data['difficulty'])

    total_steps = 2000
    gaussian_sampler = GaussianCurriculumSampler(
        difficulties, total_steps=total_steps, mu_0=0, sigma=0.3
    )

    new_difficulties = []
    samples = []

    for i in range(total_steps):
        probs = gaussian_sampler.compute_probabilities(i + 1)
        # Sample two examples at each step
        for j in range(2):
            sample_index = np.random.choice(len(difficulties), p=probs)
            new_difficulties.append(difficulties[sample_index])
            samples.append(citynavData[sample_index])

    with open("data/gaussian_samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    main()