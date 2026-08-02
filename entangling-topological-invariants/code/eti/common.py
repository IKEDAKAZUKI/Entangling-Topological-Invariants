from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(a, np.kron(b, c))


def block_diag(*mats: np.ndarray) -> np.ndarray:
    n = sum(m.shape[0] for m in mats)
    out = np.zeros((n, n), dtype=complex)
    pos = 0
    for matrix in mats:
        stop = pos + matrix.shape[0]
        out[pos:stop, pos:stop] = matrix
        pos = stop
    return out


def spectral_derivative(arr: np.ndarray, axis: int) -> np.ndarray:
    n = arr.shape[axis]
    modes = np.fft.fftfreq(n, d=1 / n)
    shape = [1] * arr.ndim
    shape[axis] = n
    modes = modes.reshape(shape)
    return np.fft.ifft(1j * modes * np.fft.fft(arr, axis=axis), axis=axis)
