import numpy as np
import sys


def __k_iterations(z: np.complex128, c: np.complex128, k_iterations: int = 0) -> np.int32:
    """Iterate the linear mapping for the mandelbrot set exactly k times

    This is used when computing the convergence scheme

     Parameters
    ----------
    z : complex
        Initial z value
    c : complex
        Intial c value
    k_iterations: int, optional (default=100)
        The number of iterations to bail out at

    Returns
    -------
    int
        The number of iterations it takes for z to become unbounded, up to max_iter
    """
    for _ in range(k_iterations):
        z[:] = z * z + c
    return z


def __iteration(z: np.complex128, c: np.complex128, max_iterations: int = 100, z_k: np.complex128 = None, epsilon: float = None) -> np.int32:
    """Iterate the linear mapping for the mandelbrot set

     Parameters
    ----------
    z : np.complex128
        Initial z value
    c : np.complex128
        Intial c value
    max_iterations: int, optional (default=100)
        The number of iterations to bail out at
    z_k : np.complex128, optional (default=None)
        z pre-run k steps forward, only needed when doing a convergence check
    epsilon : float, optional (default=None)
        the convergence epsilon |z_k - z| < epsilon

    Returns
    -------
    int
        The number of iterations it takes for z to become unbounded, up to max_iter
    """

    iterations = np.zeros_like(c, dtype=np.int32)
    mask = np.abs(z) <= 2,
    iter = 0
    while np.any(mask) and iter < max_iterations:
        z[mask] = z[mask] * z[mask] + c[mask]
        iterations[mask] += 1
        mask = np.abs(z) <= 2,
        iter += 1
    return iterations


def mandelbrot(c_values: np.ndarray, max_iterations: int = 100, k_iterations: int = None, epsilon: float = None) -> np.int32:
    """Compute the mandelbrot set

     Parameters
    ----------
    c_values : np.ndarray
        All c values we will iterate the mandelbrot map for
    max_iterations: int, optional (default=100)
        The number of iterations to bail out at
    k_iterations: int, optional (default=None)
        Supplied when computing the convergence scheme, this is the number of iterations to continue
        computing the linear mapping for. When supplied, the return value will be 2 ndarrays
    epsilon: float, optional (default=None)
        The epsilon check to use for convergence, when k_iterations is supplied

    Returns
    -------
    np.ndarray
        The number of iterations for each c value where z became unbounded
    """
    rows = c_values.shape[0]
    cols = c_values.shape[1]
    counts = np.zeros_like(c_values, dtype=np.int32)
    z = np.zeros_like(c_values, dtype=np.complex128)

    # iterate z once to generate the mandelbrot set
    for row in range(rows):
        counts[row] = __iteration(z[row], c_values[row], max_iterations=max_iterations)

    # for convergence coloring
    if k_iterations:
        k_band = 8
        # pixels represents everything in the mandelbrot set, those points that didn't diverge
        pixels = np.abs(z) < 2
        # we need a second copy of z to compute k iterations ahead of z
        zk = []
        for k in range(k_iterations+k_band):
            z_k = np.zeros_like(z, dtype=np.complex128)
            # then lead z by k iterations to obtain z_k, but only for points that didn't diverge
            for row in range(rows):
                mask = pixels[row]
                z_k[row][mask] = __k_iterations(z_k[row][mask], c_values[row][mask], k_iterations=k+1)
            zk.append(z_k)
        
        z[:] = 0
        # we need to carry three pieces of information
        # 1. points on the mandelbrot set and don't converge or diverge (0)
        #   - "Note that the Julia set (which is the boundary of the filled-in Julia set) is approximated by the set of all pixels (i, j) on the canvas such that c(i, j) = M and d(i, j) = M, which is a region of interest in chaos theory."
        #   - ^ from https://www.sekinoworld.com/fractal/coloring.htm
        #   - this approximates the boundary
        # 2. points on the mandelbrot set that do converge (positive number, when we first detect convergence)
        # 3. points that diverge (-1)
        # everything that diverged becomes negative
        counts[~pixels] = -1
        for row in range(rows):
            iter = 1
            pixel_mask = pixels[row]
            _z = z[row][pixel_mask]
            _c_values = c_values[row][pixel_mask]
            _counts = counts[row][pixel_mask]
            while iter <= max_iterations:
                _z = _z * _z + _c_values
                for k in range(k_iterations+k_band):
                    zk[k][row][pixel_mask] = zk[k][row][pixel_mask] * zk[k][row][pixel_mask] + _c_values
                    mask = (np.abs(zk[k][row][pixel_mask] - _z) < epsilon) & ((_counts == max_iterations) | (_counts > k + 1))
                    _counts[mask] = k+1
                iter += 1
            counts[row][pixel_mask] = _counts
        boundary = pixels & (counts == max_iterations)
        counts[counts > k_iterations] = -1
        counts[boundary] = 0

    return counts


def mandelbrot_threaded(c_values: np.ndarray, max_iterations: int = 100, n_workers: int = 8, k: int = None) -> np.int32:
    """Compute the mandelbrot set in parallel

    This requires free-threaded python (python>=3.14t)

    Parameters
    ----------
    c_values : np.ndarray
        All c values we will iterate the mandelbrot map for
    max_iterations: int, optional (default=100)
        The number of iterations to bail out at
    n_workers: int, optinal (default=8)
        The number of workers to run in parallel

    Returns
    -------
    np.ndarray
        The number of iterations for each c value where z became unbounded
    """

    if sys._is_gil_enabled():
        raise RuntimeError("Free-threaded python is required")

    from concurrent.futures import ThreadPoolExecutor

    rows = c_values.shape[0]
    cols = c_values.shape[1]
    counts = np.zeros((rows, cols), dtype=np.int32)
    z = np.zeros((rows, cols), dtype=np.complex128)

    if rows < n_workers:
        n_workers = rows

    def work(row_chunks):
        nonlocal counts
        start, end = row_chunks
        for row in range(start, min(end, rows)):
            counts[row] = __iteration(z[row], c_values[row], max_iterations=max_iterations)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        div = rows // n_workers
        chunks = list(range(0, rows + div, div))
        row_chunks = [(start, end) for start, end in zip(chunks[:-1], chunks[1:])]
        list(pool.map(work, row_chunks))

    return z, counts
