# fftconvolve_benchmark.py

import time
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from microsim import vkfft

MODE = "same"
TOL = 1e-5


def numpy_fftconvolve(signal: Any, kernel: Any) -> Any:
    """Perform FFT convolution using numpy."""
    return fftconvolve(signal, kernel, mode=MODE)


def torch_fftconvolve(signal: Any, kernel: Any) -> Any:
    """Perform FFT convolution using PyTorch."""
    from microsim import _torch_conv

    return _torch_conv.torch_conv(signal, kernel, mode=MODE, device="mps")


def cupy_fftconvolve(signal: Any, kernel: Any) -> Any:
    """Perform FFT convolution using PyTorch."""
    import cupyx.scipy.signal as cuf

    return cuf.fftconvolve(signal, kernel, mode=MODE)


def jax_fftconvolve(signal: Any, kernel: Any) -> Any:
    """Perform FFT convolution using JAX, JIT-compiled."""
    import jax.scipy.signal as jsig

    return jsig.fftconvolve(signal, kernel, mode=MODE)


def vk_fftconvolve(signal: Any, kernel: Any) -> Any:
    return vkfft.fftconvolve(signal, kernel, mode=MODE)


def benchmark(function: Callable, *args: Any, num_trials: int = 5) -> Any:
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
    # mi, ma, me, sh = result.min(), result.max(), result.mean(), result.shape
    # print(f"{mi=} {ma=} {me=} {sh=}")
    return result


def main() -> None:
    np.random.seed(0)

    # Example signals
    np_signal = np.random.random((9, 77, 77)).astype(np.float32)
    np_kernel = np.random.random((9, 77, 77)).astype(np.float32)

    # Benchmark numpy
    np_result = benchmark(numpy_fftconvolve, np_signal, np_kernel)

    # Benchmark JAX
    try:
        import jax.numpy as jnp
    except ImportError:
        pass
    else:
        jax_signal = jnp.asarray(np_signal)
        jax_kernel = jnp.asarray(np_kernel)
        jax_result = benchmark(jax_fftconvolve, jax_signal, jax_kernel)
        if not np.allclose(np_result, jax_result, rtol=TOL, atol=np_result.max() * TOL):
            print("  !!!!!!result does not match NumPy result")

    # Benchmark PyTorch
    try:
        import torch
    except ImportError:
        pass
    else:
        print("benchmarking torch_fftconvolve")
        torch_signal = torch.asarray(np_signal)
        torch_kernel = torch.asarray(np_kernel)
        torch_signal.requires_grad_(False)
        torch_kernel.requires_grad_(False)
        t_result = benchmark(torch_fftconvolve, torch_signal, torch_kernel)
        if not np.allclose(np_result, t_result, rtol=TOL, atol=np_result.max() * TOL):
            print("  !!!!!!result does not match NumPy result")

    # Benchmark CuPy
    try:
        import cupy
    except ImportError:
        pass
    else:
        cupy_signal = cupy.asarray(np_signal)
        cupy_kernel = cupy.asarray(np_kernel)
        cu_result = benchmark(cupy_fftconvolve, cupy_signal, cupy_kernel)
        if not np.allclose(np_result, cu_result, rtol=TOL, atol=np_result.max() * TOL):
            print("  !!!!!!result does not match NumPy result")

    try:
        import pyopencl as cl
        import pyopencl.array as cla
    except ImportError:
        pass
    else:
        cl_ctx = cl.create_some_context(interactive=False)
        cq = cl.CommandQueue(cl_ctx)

        cl_signal = cla.to_device(cq, np_signal)
        cl_kernel = cla.to_device(cq, np_kernel)
        vk_result = benchmark(vk_fftconvolve, cl_signal, cl_kernel)
        if not np.allclose(np_result, vk_result, rtol=TOL, atol=np_result.max() * TOL):
            print("  !!!!!!result does not match NumPy result")
    plt.show()


if __name__ == "__main__":
    main()
