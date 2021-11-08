
import math

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as lse
from jax import random

def log_softmax(x, axis=-1):
    Z = lse(x, axis=axis, keepdims=True)
    return x - Z

def struct2flat(x):
    batch, time = x.shape
    assert time == 2
    return x[:,0] * 2 + x[:,1]

def flat2struct(x):
    batch, = x.shape
    y = np.zeros((batch, 2))
    y[:,0] = x // 2
    y[:,1] = x % 2
    return jax.device_put(y)

class DummyStudent:
    def __init__(self, T=2, V=2):
        self.T = T
        self.V = V
        #self.unnormalized_log_probs = jnp.zeros((T, V))
        self.unnormalized_log_probs = jax.device_put(np.random.randn(T, V))
    def log_probs(self):
        Z = lse(self.unnormalized_log_probs, 1, keepdims=True)
        return self.unnormalized_log_probs - Z
    def score(self, x):
        log_probs = self.log_probs()
        return log_probs[np.arange(self.T), x]

    @staticmethod
    def compute_log_probs_struct(log_probs):
        # 00, 01, 10, 11
        return jnp.hstack((
            log_probs[0,0] + log_probs[1,0],
            log_probs[0,0] + log_probs[1,1],
            log_probs[0,1] + log_probs[1,0],
            log_probs[0,1] + log_probs[1,1],
        ))

    @staticmethod
    def compute_mle(marginals):
        log_p_newdata = marginals
        # 00, 01, 10, 11
        theta00 = jnp.logaddexp(log_p_newdata[0], log_p_newdata[1])
        theta01 = jnp.logaddexp(log_p_newdata[2], log_p_newdata[3])
        theta10 = jnp.logaddexp(log_p_newdata[0], log_p_newdata[2])
        theta11 = jnp.logaddexp(log_p_newdata[1], log_p_newdata[3])
        return jnp.vstack((
            jnp.hstack((theta00, theta01)),
            jnp.hstack((theta10, theta11)),
        ))

class EditModel:
    def __init__(self, T=2, V=2):
        self.T = T
        self.V = V
        #self.unnormalized_log_probs = jnp.zeros((T, V, V))
        # self.lp: time x oldvocab x newvocab 
        #self.unnormalized_log_probs = jax.device_put(np.random.randn(T, V, V))
        # self.lp: old(time * vocab) x new(time * vocab)
        #self.unnormalized_log_probs = jax.device_put(np.random.randn(T*V, T*V))
        #self.unnormalized_log_probs = jax.device_put(np.random.randn(T*V, T*V))
        self.unnormalized_log_probs = jnp.zeros((T*V, T*V))
        fillval = -10
        self.unnormalized_log_probs = self.unnormalized_log_probs.at[1, 0].set(fillval)
        self.unnormalized_log_probs = self.unnormalized_log_probs.at[1,-1].set(fillval)
        self.unnormalized_log_probs = self.unnormalized_log_probs.at[2, 0].set(fillval)
        self.unnormalized_log_probs = self.unnormalized_log_probs.at[2,-1].set(fillval)

        #self.unnormalized_log_probs = self.unnormalized_log_probs.at[1,1].set(0)
        #self.unnormalized_log_probs = self.unnormalized_log_probs.at[1,2].set(0)
        #self.unnormalized_log_probs = self.unnormalized_log_probs.at[2,1].set(0)
        #self.unnormalized_log_probs = self.unnormalized_log_probs.at[2,2].set(0.1)

        self.key = random.PRNGKey(0)

    def log_probs(self):
        Z = lse(self.unnormalized_log_probs, 1, keepdims=True)
        return self.unnormalized_log_probs - Z

    def score(self, data, newdata):
        return self.log_probs()[data, newdata]

    def sample(self, data):
        log_probs = self.log_probs()
        sample_log_probs = log_probs[np.arange(self.T), data]
        self.key, key = random.split(self.key)
        return random.categorical(key, sample_log_probs, axis=-1)

    def marginals(self, log_b):
        editor_log_probs = self.log_probs()
        log_q_newdata_data = log_b + editor_log_probs
        log_q_newdata = lse(log_q_newdata_data, axis=0)
        return log_q_newdata

    def log_probs(self):
        Z = lse(self.unnormalized_log_probs, 1, keepdims=True)
        return self.unnormalized_log_probs - Z

    def score(self, data, newdata):
        return self.log_probs()[data, newdata]

    def sample(self, data):
        log_probs = self.log_probs()
        sample_log_probs = log_probs[np.arange(self.T), data]
        self.key, key = random.split(self.key)
        return random.categorical(key, sample_log_probs, axis=-1)

    @staticmethod
    def compute_marginals(unnormalized_log_probs, log_b):
        Z = lse(unnormalized_log_probs, 1, keepdims=True)
        editor_log_probs = unnormalized_log_probs - Z
        log_q_newdata_data = log_b[:,None] + editor_log_probs
        log_q_newdata = lse(log_q_newdata_data, axis=0)
        return log_q_newdata

def kl(log_p, log_q):
    mask = log_p > float("-inf")
    p = jnp.exp(log_p[mask])
    return jnp.sum((p * (log_p[mask] - log_q[mask])))

def compute_objective(unnormalized_log_probs, data, editor, student):
    marginals = editor.compute_marginals(unnormalized_log_probs, data)
    mle = student.compute_mle(marginals)
    #kl_p = kl_prior(data, marginals)
    #kl_s = kl_student(marginals, student.compute_log_probs_struct(mle))
    kl_p = kl(data, marginals)
    kl_s = kl(marginals, student.compute_log_probs_struct(mle))
    return kl_p + kl_s

def generate_data(b, N=10, T=2, V=2):
    data = np.array([[0,0], [0, 1], [1, 0], [1,1]])
    idx = np.random.choice(data.shape[0], p=np.array(b), size=(N,))
    return data[idx]

def train(b, editor, student, iters, alpha):
    params = editor.unnormalized_log_probs
    kls = []
    grad_norms = []
    for i in range(iters):
        grad_fn = jax.value_and_grad(compute_objective)
        kl, grad = grad_fn(params, b, editor, student)
        params -= alpha * grad
        kls.append(kl)
        grad_norms.append(jnp.linalg.norm(grad))
    return params, kls, grad_norms

def plot(kls, grad_norms, prefix=""):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    iters = np.arange(len(kls))
    df = pd.DataFrame(
        np.vstack((iters, kls, grad_norms)).T,
        columns = ["iters", "kl", "gradnorm"],
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
    g0 = sns.lineplot(
        data = df, x = "iters", y = "kl", ax=ax0,
    )
    ax0.set_xlabel("Iters")
    ax0.set_ylabel("KL")

    g1 = sns.lineplot(
        data = df, x = "iters", y = "gradnorm", ax=ax1,
    )
    ax1.set_xlabel("Iters")
    ax1.set_ylabel("Grad Norm")

    fig.tight_layout()
    fig.savefig(f"plots/{prefix}training_plots.png")
    plt.close()

def plot_heatmap(im, filename, labels):
    import matplotlib.pyplot as plt
    import seaborn as sns

    ax = sns.heatmap(im, vmin=0, vmax=1, square=True,
        xticklabels = labels,
        yticklabels = labels,
    )
    plt.savefig(f"plots/{filename}.png")
    plt.close()


def run(B, iters=1000, alpha=1e-1, do_plot=False, plot_string=""):
    print("True dist params")
    print(B)
    log_B = jnp.log(B)
    student = DummyStudent()
    editor = EditModel()

    # initial student params
    marginals = editor.compute_marginals(editor.unnormalized_log_probs, log_B)
    student_params = jnp.exp(log_softmax(student.compute_mle(marginals)))
    editor_params = jnp.exp(log_softmax(editor.unnormalized_log_probs))
    print("Initial editor params")
    print(editor_params)
    print("Initial student params")
    print(student_params)
    editor_labels = ["00", "01", "10", "11"]
    student_labels = ["0", "1"]
    if do_plot:
        plot_heatmap(editor_params, f"{plot_string}editor_init", editor_labels)
        plot_heatmap(student_params, f"{plot_string}student_init", student_labels)

    params, kls, grad_norms = train(log_B, editor, student, iters=iters, alpha=alpha)

    editor.unnormalized_log_probs = params
    marginals = editor.compute_marginals(params, log_B)
    student_params = jnp.exp(log_softmax(student.compute_mle(marginals)))
    editor_params = jnp.exp(log_softmax(params))
    print("Initial editor params")
    print(editor_params)
    print("Final student params")
    print(student_params)
    if do_plot:
        plot_heatmap(editor_params, f"{plot_string}editor_final", editor_labels)
        plot_heatmap(student_params, f"{plot_string}student_final", student_labels)
        plot(kls, grad_norms, plot_string)


def main():
    do_plot = True

    pad = 1e-3
    low = 0.45
    B = jnp.array([pad, low, 1-2*pad-low, pad])
    print("VERY UNEVEN PRIOR")
    run(B, do_plot=do_plot, plot_string="very_uneven_prior_")

    pad = 1e-3
    low = 0.49
    B = jnp.array([pad, low, 1-2*pad-low, pad])
    print("UNEVEN PRIOR")
    run(B, do_plot=do_plot, plot_string="uneven_prior_")

    low = (1 - 2 * pad) / 2
    B = jnp.array([pad, low, 1-2*pad-low, pad])
    print("EVEN PRIOR")
    run(B, do_plot=do_plot, plot_string="even_prior_")


if __name__ == "__main__":
    main()
