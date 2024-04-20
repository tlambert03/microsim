# fftconvolve_benchmark.py

import time
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import jax.scipy.signal as jsig
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from microsim import vkfft

MODE = "same"


def numpy_fftconvolve(signal, kernel):
    """Perform FFT convolution using numpy."""
    return fftconvolve(signal, kernel, mode=MODE)


def torch_fftconvolve(signal, kernel):
    """Perform FFT convolution using PyTorch."""
    import torchaudio.functional as taf

    return taf.fftconvolve(signal, kernel, mode=MODE)


def cupy_fftconvolve(signal, kernel):
    """Perform FFT convolution using PyTorch."""
    import cupyx.scipy.signal as cuf

    return cuf.fftconvolve(signal, kernel, mode=MODE)


def jax_fftconvolve(signal, kernel):
    """Perform FFT convolution using JAX, JIT-compiled."""
    return jsig.fftconvolve(signal, kernel, mode=MODE)


def vk_fftconvolve(signal, kernel):
    return vkfft.fftconvolve(signal, kernel, mode=MODE)


def benchmark(function: Callable, *args: Any, num_trials: int = 5) -> None:
    """Benchmark a given FFT convolution function."""
    _ = function(*args)  # Warmup
    times = []
    print("benchmarking", function)
    for _ in range(num_trials):
        start_time = time.time()
        result = function(*args)
        end_time = time.time()
        times.append(end_time - start_time)
    if hasattr(result, "get"):
        result = result.get()
    plt.figure()
    plt.imshow(result[result.shape[0] // 2])
    avg_time = sum(times) / num_trials
    print(f"Avg {avg_time:.6f} seconds,  min: {min(times):.6f} seconds")
    print("L2 norm:", np.linalg.norm(result))
    mi, ma, me, sh = result.min(), result.max(), result.mean(), result.shape
    print(f"{mi=} {ma=} {me=} {sh=}")


def main() -> None:
    np.random.seed(0)
    from skimage import data

    # Example signals
    # np_signal = np.random.random((32, 128, 128)).astype(np.float32)
    np_signal = data.cells3d()[:, 1].astype(np.float32)
    np_kernel = np.zeros(np_signal.shape).astype(np.float32)
    np_kernel[32, 128, 128] = 1
    np_kernel[32, 120, 120] = 1
    # Benchmark numpy
    benchmark(numpy_fftconvolve, np_signal, np_kernel)

    # Benchmark JAX
    jax_signal = jnp.asarray(np_signal)
    jax_kernel = jnp.asarray(np_kernel)
    benchmark(jax_fftconvolve, jax_signal, jax_kernel)

    # Benchmark PyTorch
    try:
        import torch
    except ImportError:
        pass
    else:
        torch_signal = torch.asarray(np_signal)
        torch_kernel = torch.asarray(np_kernel)
        benchmark(torch_fftconvolve, torch_signal, torch_kernel)

    # Benchmark CuPy
    try:
        import cupy
    except ImportError:
        pass
    else:
        cupy_signal = cupy.asarray(np_signal)
        cupy_kernel = cupy.asarray(np_kernel)
        benchmark(cupy_fftconvolve, cupy_signal, cupy_kernel)

    try:
        import pyopencl as cl
        import pyopencl.array as cla
    except ImportError:
        pass
    else:
        cl_ctx = cl.create_some_context(interactive=False)
        cq = cl.CommandQueue(cl_ctx)

        cl_signal = cla.to_device(cq, np_signal)
        cl_kernel = cla.to_device(cq, np.fft.fftshift(np_kernel))
        benchmark(vk_fftconvolve, cl_signal, cl_kernel)

    plt.show()


if __name__ == "__main__":
    main()
