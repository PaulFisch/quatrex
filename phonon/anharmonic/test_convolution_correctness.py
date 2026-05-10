"""Verify convolution correctness on different grids.

Tests with a known analytic function: convolve two Gaussians.
The exact result is known, so we can verify:
  1. Positive-only FFT convolution (OLD code approach)
  2. Symmetric grid FFT convolution (NEW code approach)
  3. Direct summation (reference)

For two Gaussians g(f) = exp(-f^2/(2σ^2)):
  (g * g)(f) = σ√π * exp(-f^2/(4σ^2))   [continuous, full integral -∞ to ∞]

But on a positive-only grid, the FFT convolution only gives:
  ∫_0^∞ g(f') g(f-f') df'   [HALF the full integral due to even symmetry]
"""
import numpy as np

sigma = 3.0  # THz


def gaussian(f):
    return np.exp(-0.5 * f**2 / sigma**2)


def exact_full_convolution(f):
    """Exact ∫_{-∞}^{∞} g(f') g(f-f') df'."""
    return sigma * np.sqrt(np.pi) * np.exp(-f**2 / (4 * sigma**2))


# --- Grid 1: Positive-only [0, 15] ---
N_pos = 101
fmax = 15.0
freqs_pos = np.linspace(0, fmax, N_pos)
dw_pos = freqs_pos[1] - freqs_pos[0]
g_pos = gaussian(freqs_pos)

# FFT convolution (old code approach: zero-pad, no even extension)
n_fft_pos = 2 * N_pos - 1
g_pad_pos = np.zeros(n_fft_pos)
g_pad_pos[:N_pos] = g_pos
C_pos_fft = np.fft.ifft(np.fft.fft(g_pad_pos)**2).real[:N_pos] * dw_pos

# Direct sum for verification
C_pos_direct = np.zeros(N_pos)
for n in range(N_pos):
    for k in range(N_pos):
        m = n - k
        if 0 <= m < N_pos:
            C_pos_direct[n] += g_pos[k] * g_pos[m] * dw_pos

# --- Grid 2: Positive-only with even extension (handles G(-f)=G(f)) ---
g_even = np.zeros(n_fft_pos)
g_even[:N_pos] = g_pos
g_even[N_pos:2*N_pos-2] = g_pos[-2:0:-1]  # mirror excluding both endpoints
C_even_fft = np.fft.ifft(np.fft.fft(g_pad_pos) * np.fft.fft(g_even)).real[:N_pos] * dw_pos

# --- Grid 3: Symmetric [-15, 15] (new code approach) ---
freqs_sym = np.concatenate((-freqs_pos[:0:-1], freqs_pos))
n_sym = len(freqs_sym)
dw_sym = freqs_sym[1] - freqs_sym[0]
g_sym = gaussian(freqs_sym)
mid = (n_sym - 1) // 2

n_fft_sym = 2 * n_sym - 1
g_pad_sym = np.zeros(n_fft_sym)
g_pad_sym[:n_sym] = g_sym
C_sym_fft = np.fft.ifft(np.fft.fft(g_pad_sym)**2).real
freq_sl = slice(mid, mid + n_sym)
C_sym = C_sym_fft[freq_sl] * dw_sym

# Extract positive part of symmetric result
C_sym_pos = C_sym[mid:]

# Exact result at positive frequencies
C_exact = exact_full_convolution(freqs_pos)

# --- Print comparison ---
print("Convolution test: g*g for Gaussian g(f) = exp(-f^2/(2*3^2))")
print(f"Exact full-range result: σ√π * exp(-f^2/(4σ^2))")
print(f"Grid: [0, {fmax}], N={N_pos}, dw={dw_pos:.4f}\n")

print(f"{'f (THz)':>8} {'Exact':>12} {'Pos FFT':>12} {'Even FFT':>12} {'Sym FFT':>12} {'Pos Direct':>12}")
print("-" * 72)
for i in range(0, N_pos, 10):
    f = freqs_pos[i]
    print(f"{f:8.2f} {C_exact[i]:12.4f} {C_pos_fft[i]:12.4f} {C_even_fft[i]:12.4f} "
          f"{C_sym_pos[i]:12.4f} {C_pos_direct[i]:12.4f}")

print(f"\nMax relative error vs exact full integral:")
nz = C_exact > 1e-6
print(f"  Pos-only FFT (old code):   {np.max(np.abs(C_pos_fft[nz] - C_exact[nz]) / C_exact[nz]):.4f}")
print(f"  Even-extension FFT:        {np.max(np.abs(C_even_fft[nz] - C_exact[nz]) / C_exact[nz]):.4f}")
print(f"  Symmetric grid FFT (new):  {np.max(np.abs(C_sym_pos[nz] - C_exact[nz]) / C_exact[nz]):.4f}")
print(f"  Pos-only direct sum:       {np.max(np.abs(C_pos_direct[nz] - C_exact[nz]) / C_exact[nz]):.4f}")

print(f"\nAt f=0:")
print(f"  Exact:      {C_exact[0]:.4f}")
print(f"  Pos FFT:    {C_pos_fft[0]:.4f}  (ratio: {C_pos_fft[0]/C_exact[0]:.4f})")
print(f"  Even FFT:   {C_even_fft[0]:.4f}  (ratio: {C_even_fft[0]/C_exact[0]:.4f})")
print(f"  Sym FFT:    {C_sym_pos[0]:.4f}  (ratio: {C_sym_pos[0]/C_exact[0]:.4f})")
print(f"  Pos direct: {C_pos_direct[0]:.4f}  (ratio: {C_pos_direct[0]/C_exact[0]:.4f})")

print(f"\nAt f=5:")
i5 = np.argmin(np.abs(freqs_pos - 5.0))
print(f"  Exact:      {C_exact[i5]:.4f}")
print(f"  Pos FFT:    {C_pos_fft[i5]:.4f}  (ratio: {C_pos_fft[i5]/C_exact[i5]:.4f})")
print(f"  Even FFT:   {C_even_fft[i5]:.4f}  (ratio: {C_even_fft[i5]/C_exact[i5]:.4f})")
print(f"  Sym FFT:    {C_sym_pos[i5]:.4f}  (ratio: {C_sym_pos[i5]/C_exact[i5]:.4f})")

# Key insight: Pos-only FFT captures ∫_0^∞ g(f') g(f-f') df' ONLY for f-f' >= 0
# i.e., ONLY the k <= n terms. For k > n, g(f-f') at negative argument is treated as 0.
# This misses roughly half the integral.
print(f"\n--- Summary ---")
print(f"Pos-only FFT / Exact at f=0: {C_pos_fft[0]/C_exact[0]:.4f}")
print(f"  → Old code captured ~{C_pos_fft[0]/C_exact[0]*100:.0f}% of the full convolution")
print(f"Symmetric FFT / Exact at f=0: {C_sym_pos[0]/C_exact[0]:.4f}")
print(f"  → New code captures ~{C_sym_pos[0]/C_exact[0]*100:.0f}% of the full convolution")
