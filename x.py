# fftconvolve_benchmark.py

import time

import jax.numpy as jnp
import jax.scipy.signal as jsig
import numpy as np
import torch
import torchaudio.functional as taf
from scipy.signal import fftconvolve


def numpy_fftconvolve(signal, kernel):
    """Perform FFT convolution using numpy."""
    return fftconvolve(signal, kernel, mode="same")


def torch_fftconvolve(signal, kernel):
    """Perform FFT convolution using PyTorch."""
    return taf.fftconvolve(signal, kernel, mode="same")


def jax_fftconvolve(signal, kernel):
    """Perform FFT convolution using JAX, JIT-compiled."""
    return jsig.fftconvolve(signal, kernel, mode="same")


def benchmark(function, *args, num_trials=10):
    """Benchmark a given FFT convolution function."""
    start_time = time.time()
    for _ in range(num_trials):
        _ = function(*args)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_trials
    return avg_time


def main():
    # Example signals
    np_signal = np.random.random((256, 1024, 1024)).astype(np.float32)
    np_kernel = np.random.random((256, 1024, 1024)).astype(np.float32)

    # Benchmark numpy
    numpy_time = benchmark(numpy_fftconvolve, np_signal, np_kernel)
    print(f"Numpy FFT Convolve: {numpy_time:.6f} seconds")

    # Benchmark JAX
    jax_signal = jnp.array(np_signal, dtype=jnp.float32)
    jax_kernel = jnp.array(np_kernel, dtype=jnp.float32)
    jax_time = benchmark(jax_fftconvolve, jax_signal, jax_kernel)
    print(f"JAX FFT Convolve: {jax_time:.6f} seconds")

    # Benchmark PyTorch
    torch_signal = torch.tensor(np_signal, dtype=torch.float32)
    torch_kernel = torch.tensor(np_kernel, dtype=torch.float32)
    torch_time = benchmark(torch_fftconvolve, torch_signal, torch_kernel)
    print(f"PyTorch FFT Convolve: {torch_time:.6f} seconds")


if __name__ == "__main__":
    main()
