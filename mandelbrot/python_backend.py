import numpy as np
import sys


def __iteration(z: complex, c: complex, max_iterations: int = 100) -> np.int32:
    """Iterate the linear mapping for the mandelbrot set

     Parameters
    ----------
    z : complex
        Initial z value
    c : complex
        Intial c value
    max_iterations: int, optional (default=100)
        The number of iterations to bail out at

    Returns
    -------
    int
        The number of iterations it takes for z to become unbounded, up to max_iter
    """
    iterations = np.int32(0)
    while np.abs(z) <= 2 and iterations < max_iterations:
        z = z**2 + c
        iterations += 1
    return iterations


def mandelbrot(c_vals: np.ndarray, max_iterations: int = 100) -> np.int32:
    """Compute the mandelbrot set

     Parameters
    ----------
    c_vals : np.ndarray
        All c values we will iterate the mandelbrot map for
    max_iterations: int, optional (default=100)
        The number of iterations to bail out at

    Returns
    -------
    np.ndarray
        The number of iterations for each c value where z became unbounded
    """
    rows = c_vals.shape[0]
    cols = c_vals.shape[1]
    counts = np.zeros((rows, cols), dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            counts[i][j] = __iteration(0, c_vals[i][j], max_iterations=max_iterations)

    return counts


def mandelbrot_threaded(c_vals: np.ndarray, max_iterations: int = 100, n_workers = 12) -> np.int32:
    """Compute the mandelbrot set in parallel

    This requires free-threaded python (python>=3.14t)

    Parameters
    ----------
    c_vals : np.ndarray
        All c values we will iterate the mandelbrot map for
    max_iterations: int, optional (default=100)
        The number of iterations to bail out at
    n_workers: int, optinal (default=12)
        The number of workers to run in parallel

    Returns
    -------
    np.ndarray
        The number of iterations for each c value where z became unbounded
    """

    if sys._is_gil_enabled():
        raise RuntimeError("Free-threaded python is required")
    
    from concurrent.futures import ThreadPoolExecutor

    rows = c_vals.shape[0]
    cols = c_vals.shape[1]
    counts = np.zeros((rows, cols), dtype=np.int32)

    def work(row_chunks):
        nonlocal counts
        start, end = row_chunks
        for row in range(start, min(end, rows)):
            for col in range(cols):
                counts[row][col] = __iteration(
                    0, c_vals[row][col], max_iterations=max_iterations
                )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        div = rows // n_workers
        chunks = list(range(0, rows + div, div))
        row_chunks = [(start, end) for start, end in zip(chunks[:-1], chunks[1:])]
        list(pool.map(work, row_chunks))

    return counts
