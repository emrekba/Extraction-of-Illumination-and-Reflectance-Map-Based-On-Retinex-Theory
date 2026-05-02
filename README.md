# Retinex-Based Illumination Map Extractor

Structure-preserving illumination map extraction using **Retinex theory** and **L1-regularized optimization** (IRLS). Designed for deepfake detection and inverse rendering applications where edge-preserving accuracy is critical.

---

## How It Works

Based on the **Retinex decomposition**:

```
L(x) = R(x) ∘ M(x)
```

where **L** is the input image, **R** is the reflectance (surface color), and **M** is the illumination map.

### Algorithm Steps

**Step 1 — Initial estimate (Eq. 2)**  
The initial illumination map is computed as the per-pixel maximum across RGB channels:
```
M̂(x) = max{ L^R(x), L^G(x), L^B(x) }
```

**Step 2 — Weight matrix (Eq. 4)**  
Sobel operators are applied to M̂ to produce a single gradient magnitude image, which defines the edge-aware weight matrix:
```
|∇M̂|(x) = √( Gx(x)² + Gy(x)² )
W(x)    = 1 / ( |∇M̂|(x) + ε )
```
W is small at edges (preserve them) and large in flat regions (smooth aggressively).

**Step 3 — Structure-preserving optimization (Eq. 3)**  
The refined illumination map M is obtained by solving:
```
min_M  ‖M̂ − M‖²_F  +  α · ‖W ∘ ∇M‖₁
```
The L1 norm is solved via **Iteratively Reweighted Least Squares (IRLS)**. Each iteration solves:
```
(I + α · L_W) m = m̂
```
where `L_W = Dx^T Ω Dx + Dy^T Ω Dy` is the weighted graph Laplacian and `spsolve` from `scipy.sparse.linalg` is used as the solver.

**Step 4 — Reflectance**  
```
R(x) = L(x) / (M(x) + ε)
```

---

## Installation

```bash
pip install numpy opencv-python scipy
```

---

## Usage

```bash
python retinex_msr.py <input> [options]
```

**Positional arguments:**

| Argument | Description |
|---|---|
| `input` | Path to a single image file or a folder of images |

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--out` | `./output` | Output directory |
| `--alpha` | `0.5` | Smoothing strength — see [Choosing Alpha](#choosing-alpha) |
| `--iters` | `10` | Maximum number of IRLS iterations |
| `--ext` | `.png` | Output file extension (`.png`, `.jpg`, etc.) |
| `--no-reflectance` | — | If set, only the illumination map is saved |

**Examples:**

```bash
# Single image with defaults
python retinex_msr.py photo.jpg

# Single image with custom alpha
python retinex_msr.py photo.jpg --alpha 2.0

# Entire folder, illumination maps only
python retinex_msr.py ./dataset --out ./results --alpha 1.5 --no-reflectance

# Custom output format
python retinex_msr.py photo.jpg --out ./out --ext .jpg --iters 15
```

**Outputs saved to `--out`:**

| File | Description |
|---|---|
| `<name>-illumination.png` | Illumination map M — grayscale, single channel |
| `<name>-reflectance.png` | Reflectance map R — 3-channel RGB with gamma correction |

---

## Choosing Alpha

| Alpha | Effect |
|---|---|
| `< 0.3` | Too small — fine textures (hair, skin pores) still visible |
| `0.5 – 1.0` | Moderate smoothing — good starting point |
| `1.5 – 2.5` | Typically optimal — textures removed, large-scale lighting preserved |
| `> 5.0` | Over-smoothed — lighting gradients also start to vanish |

**What alpha controls mathematically:**

```
(I  +  α · L_W)  m  =  m̂
 ─               ─────
 data fidelity   regularization
```

- `α → 0`: solution `m ≈ m̂` (no smoothing, texture visible)
- `α → ∞`: solution `m → constant` (maximum smoothing, lighting gradients lost)
- The weight matrix W ensures edges are preserved regardless of alpha

---

## Project Structure

```
.
├── retinex_msr.py   # Illumination map extractor
└── output/          # Generated outputs
```

---

## References

- **LIDeepDet**: *Deepfake Detection via Image Decomposition and Advanced Lighting Information Analysis*, Electronics 2024, 13(22), 4466. [DOI: 10.3390/electronics13224466](https://doi.org/10.3390/electronics13224466)
- Land, E.H. & McCann, J.J. (1971). *Lightness and the Retinex Theory*. JOSA.
- Guo, X. et al. (2017). *LIME: Low-Light Image Enhancement via Illumination Map Estimation*. IEEE TIP.
