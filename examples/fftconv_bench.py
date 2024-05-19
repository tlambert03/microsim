# fftconvolve_benchmark.py

import time

try:
    import cupy
    import cupyx.scipy.signal as cuf
except ImportError:
    cupy = None

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


def cupy_fftconvolve(signal, kernel):
    """Perform FFT convolution using PyTorch."""
    return cuf.fftconvolve(signal, kernel, mode="same")


def jax_fftconvolve(signal, kernel):
    """Perform FFT convolution using JAX, JIT-compiled."""
    return jsig.fftconvolve(signal, kernel, mode="same")


def benchmark(function, *args, num_trials=10):
    """Benchmark a given FFT convolution function."""
    _ = function(*args)  # Warmup
    times = []
    print("benchmarking", function)
    for _ in range(num_trials):
        start_time = time.time()
        result = function(*args)
        end_time = time.time()
        times.append(end_time - start_time)
    print(result.min(), result.max(), result.mean(), result.shape)
    avg_time = sum(times) / num_trials
    return avg_time, min(times)


def main() -> None:
    np.random.seed(0)

    # Example signals
    np_signal = np.random.random((32, 256, 256)).astype(np.float32)
    np_kernel = np.random.random((32, 256, 256)).astype(np.float32)
    # Benchmark numpy
    numpy_time, mint = benchmark(numpy_fftconvolve, np_signal, np_kernel)
    print(f"Numpy FFT Convolve: {numpy_time:.6f} seconds, {mint:.6f} seconds")

    # Benchmark JAX
    jax_signal = jnp.asarray(np_signal)
    jax_kernel = jnp.asarray(np_kernel)
    jax_time, mint = benchmark(jax_fftconvolve, jax_signal, jax_kernel)
    print(f"JAX FFT Convolve: {jax_time:.6f} seconds, {mint:.6f} seconds")

    # Benchmark PyTorch
    torch_signal = torch.asarray(np_signal)
    torch_kernel = torch.asarray(np_kernel)
    torch_time, mint = benchmark(torch_fftconvolve, torch_signal, torch_kernel)
    print(f"PyTorch FFT Convolve: {torch_time:.6f} seconds, {mint:.6f} seconds")

    # Benchmark CuPy
    if cupy is not None:
        cupy_signal = cupy.asarray(np_signal)
        cupy_kernel = cupy.asarray(np_kernel)
        cupy_time, mint = benchmark(cupy_fftconvolve, cupy_signal, cupy_kernel)
        print(f"CuPy FFT Convolve: {cupy_time:.6f} seconds, {mint:.6f} seconds")


if __name__ == "__main__":
    main()
