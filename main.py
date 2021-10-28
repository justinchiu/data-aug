
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as lse
from jax import random

import strux


class DummyStudent:
    def __init__(self, T=2, V=2):
        self.T = T
        self.V = V
        #self.unnormalized_log_probs = jnp.zeros((T, V))
        self.unnormalized_log_probs = jax.device_put(np.random.randn(T, V))

    def score(self, x):
        Z = lse(self.unnormalized_log_probs, 1, keepdims=True)
        log_probs = self.unnormalized_log_probs - Z
        return log_probs[np.arange(self.T), x]

class EditModel:
    def __init__(self, T=2, V=2):
        self.T = T
        self.V = V
        #self.unnormalized_log_probs = jnp.zeros((T, V, V))
        # self.lp: time x oldvocab x newvocab 
        #self.unnormalized_log_probs = jax.device_put(np.random.randn(T, V, V))
        # self.lp: old(time * vocab) x new(time * vocab)
        self.unnormalized_log_probs = jax.device_put(np.random.randn(T*V, T*V))
        self.key = random.PRNGKey(0)

    def score(self, data, newdata):
        Z = lse(self.unnormalized_log_probs, 2, keepdims=True)
        log_probs = self.unnormalized_log_probs - Z
        return log_probs[np.arange(self.T), data, newdata]

    def sample(self, data):
        Z = lse(self.unnormalized_log_probs, 2, keepdims=True)
        log_probs = self.unnormalized_log_probs - Z
        sample_log_probs = log_probs[np.arange(self.T), data]
        self.key, key = random.split(self.key)
        return random.categorical(key, sample_log_probs, axis=-1)

def kl(log_p, log_q):
    return 

def generate_data(N=10, T=2, V=2):
    data = np.array([[0, 1], [1, 0]])
    idx = np.random.binomial(n=1, p=0.3, size=(N,))
    return data[idx]

def train(data, model):
    pass

def main():
    data = generate_data()
    student = DummyStudent()
    student.score(data)
    editor = EditModel()
    editor.sample(data)



if __name__ == "__main__":
    main()
