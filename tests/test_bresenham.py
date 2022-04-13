import numpy as np

from microsim.samples import drawlines_bresenham


def test_bres(benchmark):
    a: np.ndarray = np.zeros((8, 8)).astype(np.int32)
    benchmark(drawlines_bresenham, np.array([[0, 0, 5, 5]], dtype=np.int32), a)
    a = a // a.max()

    expect: np.ndarray = np.asarray(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(a, expect)


def test_bres3(benchmark):
    a: np.ndarray = np.zeros((4, 4, 4)).astype(np.int32)
    benchmark(drawlines_bresenham, np.array([[0, 0, 0, 3, 3, 3]], dtype=np.int32), a)
    a = a // a.max()

    expect: np.ndarray = np.asarray(
        [
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(a, expect)
