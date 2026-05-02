"""
Retinex Tabanli Yapi-Koruyucu Aydinlatma Haritasi Cikarici
===========================================================

Kullanimm:
    python retinex_msr.py <goruntu_yolu> [--alpha 0.1] [--out ./output] [--iters 10]

Ornek:
    python retinex_msr.py images.jpeg
    python retinex_msr.py images.jpeg --alpha 0.05 --iters 15

Algoritma (agent.md):
    1. Baslangic tahmini  : M_hat(x) = max_c { L^c(x) }   (Denklem 2)
    2. Agirlik matrisi    : W_x = 1 / (|Sobel_x(L)| + eps)  (WLS mantigi)
    3. Optimizasyon       : min_M ||M_hat - M||_F^2 + alpha * ||W o nabla M||_1
    4. Cozucu             : IRLS --> her iterasyonda sparse lineer sistem:
                             (I + alpha * L_W) m = m_hat
                             scipy.sparse.linalg.spsolve ile cozulur
"""

import io
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Windows konsolunda UTF-8 ciktisi
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ======================================================================
# ADIM 0: Yardimci — sparse fark matrisleri
# ======================================================================

def _build_Dx(H: int, W: int) -> sparse.csr_matrix:
    """
    Yatay forward-difference matrisi  D_x  (N x N, N = H*W).

    (D_x @ m)[i] = m[i+1] - m[i]  (ayni satir icindeki komsular)

    Sinir kosulu (Neumann, sifir-akis):
      - Sag kenar pikselleri (sutun W-1): hem diag_main hem diag_plus1 sifir
        => turev tamamen kapatilir, -1 kalintisi sistemi bozmaz
      - Satir gecislerinde (i*W - 1) diag_plus1 sifir (eski davranis korunur)
    """
    N = H * W
    diag_main  = -np.ones(N)
    diag_plus1 =  np.ones(N - 1)

    # Sag kenar piksellerinde ana diyagonali da sifirla (tam Neumann)
    diag_main[W - 1::W] = 0.0
    # Satir gecislerinde off-diagonal sifirla
    for i in range(1, H):
        diag_plus1[i * W - 1] = 0.0

    return sparse.diags([diag_main, diag_plus1], [0, 1],
                        shape=(N, N), format="csr")


def _build_Dy(H: int, W: int) -> sparse.csr_matrix:
    """
    Dikey forward-difference matrisi  D_y  (N x N).

    (D_y @ m)[i] = m[i+W] - m[i]  (ust-alt komsular)

    Sinir kosulu (Neumann, sifir-akis):
      - Alt kenar pikselleri (son W piksel): diag_main sifirlanir
        => alt sinirda turev tamamen kapatilir
    """
    N = H * W
    diag_main = -np.ones(N)
    diag_W    =  np.ones(N - W)

    # Alt kenar piksellerinde ana diyagonali sifirla (tam Neumann)
    diag_main[-W:] = 0.0

    return sparse.diags([diag_main, diag_W], [0, W],
                        shape=(N, N), format="csr")


# ======================================================================
# ANA FONKSIYON
# ======================================================================

def extract_illumination_map(image_path: str,
                             alpha: float = 0.5,
                             n_iters: int = 10,
                             eps: float = 1e-4,
                             irls_delta: float = 1e-6,
                             irls_tol: float = 1e-4) -> np.ndarray:
    """
    Verilen RGB goruntuden yapi-koruyucu aydinlatma haritasi cikarir.

    Parametreler
    ------------
    image_path  : str    -- Giris goruntu dosyasi yolu
    alpha       : float  -- Duzgunlestirme / kenar-koruma katsayisi (Denklem 3)
                           0.5 ideal baslangic; kucuk = daha fazla doku, buyuk = daha duzgun
    n_iters     : int    -- IRLS maks iterasyon sayisi (genellikle 5-15 yeterli)
    eps         : float  -- Agirlik hesabinda sifir koruma: W = 1/(|grad| + eps)
    irls_delta  : float  -- IRLS bolenden sifir koruma: omega = W/(|grad_M| + delta)
    irls_tol    : float  -- IRLS erken cikis esigi: ||m_new - m_old|| / ||m_old|| < tol

    Dondutur
    --------
    M : np.ndarray, shape (H, W), dtype float32, aralik [0, 1]
        Normalize edilmis aydinlatma haritasi.

    Algoritma
    ---------
    Denklem 2:  M_hat = max_c L^c(x)          baslangic tahmini
    Denklem 3:  min_M ||M_hat - M||_F^2 + alpha * ||W o nablaM||_1
    Denklem 4:  Sobel gradyanlari ile nablaM

    IRLS yaklasimlama:
        ||W o nabla M||_1  ≈  Σ_i  [W_i / (|nablaM_i|^(k) + delta)] * (nablaM_i)^2
        =>  her iterasyonda: (I + alpha * L_W^(k)) m^(k+1) = m_hat
            burada  L_W^(k) = Dx^T Omega_x^(k) Dx + Dy^T Omega_y^(k) Dy
    """

    # ------------------------------------------------------------------
    # ADIM 0: Goruntu okuma ve normalizasyon
    # ------------------------------------------------------------------
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Goruntu okunamadi: {image_path}")

    # BGR -> RGB, float64 [0, 1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    H, W, _ = rgb.shape
    N = H * W

    # ------------------------------------------------------------------
    # ADIM 1: Baslangic Aydinlatma Tahmini  (Denklem 2)
    #   M_hat(x) = max_{c in {R,G,B}} L^c(x)
    # ------------------------------------------------------------------
    M_hat = np.max(rgb, axis=2)          # shape: (H, W),  aralik [0, 1]
    m_hat = M_hat.ravel()                # shape: (N,)  -- flatten (row-major)

    # ------------------------------------------------------------------
    # ADIM 3: Agirlik Matrisi W Hesabi (Denklem 3 + Denklem 4 — makale uyumlu)
    #
    #   Denklem 4: Gx ve Gy Sobel kernelleri M_hat'e uygulanir.
    #   Makale: "...which are then COMBINED to give a SINGLE gradient
    #            magnitude image."
    #
    #   => |nabla M_hat|(x) = sqrt( Gx(x)^2 + Gy(x)^2 )   (tek buyukluk)
    #   =>  W(x) = 1 / (|nabla M_hat|(x) + eps)             (tek W matrisi)
    #
    #   Hem x hem y yonunde AYNI W kullanilir (makaledeki tek-W formul).
    #   Kenarda |nabla| buyuk => W kucuk  (kenari koru, az duzgunlestir)
    #   Duzlukte |nabla| kucuk => W buyuk (agresif duzgunlestir)
    # ------------------------------------------------------------------
    M_hat_f32 = M_hat.astype(np.float32)   # (H, W) tek kanal — max(R,G,B)

    # Sobel gradyanlari M_hat uzerinde (Denklem 4)
    gx_M = cv2.Sobel(M_hat_f32, cv2.CV_64F, 1, 0, ksize=3)   # yatay gradyan
    gy_M = cv2.Sobel(M_hat_f32, cv2.CV_64F, 0, 1, ksize=3)   # dikey gradyan

    # Birlestirme: tek gradyan buyuklugu (Denklem 4 — "single gradient magnitude image")
    grad_mag = np.sqrt(gx_M ** 2 + gy_M ** 2)                # (H, W)

    # Tek W matrisi — hem x hem y yonunde kullanilir
    W_flat = (1.0 / (grad_mag + eps)).ravel()   # (N,)  -- NOT: W = goruntu genisligi!
    Wx = W_flat
    Wy = W_flat

    # ------------------------------------------------------------------
    # ADIM 4: Sparse Fark Matrisleri ve Optimizasyon Hazirlik
    #   D_x: (N x N) yatay forward-difference
    #   D_y: (N x N) dikey  forward-difference
    # ------------------------------------------------------------------
    print(f"  Boyut: {H}x{W} ({N} piksel) | Sparse matris olusturuluyor...")
    Dx = _build_Dx(H, W)     # (N x N)
    Dy = _build_Dy(H, W)     # (N x N)

    Identity = sparse.eye(N, format="csr")

    # ------------------------------------------------------------------
    # IRLS Iterasyonlari
    #   Hedef: min_m  ||m_hat - m||^2 + alpha * ||W o (D m)||_1
    #
    #   Her iterasyonda:
    #     1. Mevcut m ile gradyan hesapla: dm_x = Dx @ m,  dm_y = Dy @ m
    #     2. IRLS agirliklari: omega_x = Wx / (|dm_x| + delta)
    #     3. Omega_x ve Omega_y ile agirlikli Laplacian: L_W = Dx^T Omega_x Dx + ...
    #     4. Normal denklem: (I + alpha * L_W) m_new = m_hat
    #     5. m = clip(m_new, 0, 1)
    # ------------------------------------------------------------------
    m = m_hat.copy()   # baslangic: M_hat ile basla

    print(f"  IRLS optimizasyonu: {n_iters} iterasyon, alpha={alpha}")
    for it in range(n_iters):

        # Mevcut m ile gradyanlar (Denklem 4 ruhunda)
        dm_x = Dx.dot(m)    # (N,) yatay fark
        dm_y = Dy.dot(m)    # (N,) dikey fark

        # IRLS agirliklari: W_i / (|nabla M_i| + delta)
        omega_x = Wx / (np.abs(dm_x) + irls_delta)   # (N,)
        omega_y = Wy / (np.abs(dm_y) + irls_delta)   # (N,)

        # Diyagonal agirlik matrisleri
        Omega_x = sparse.diags(omega_x, 0, shape=(N, N), format="csr")
        Omega_y = sparse.diags(omega_y, 0, shape=(N, N), format="csr")

        # Agirlikli Laplacian:  L_W = Dx^T Omega_x Dx + Dy^T Omega_y Dy
        L_W = (Dx.T.dot(Omega_x.dot(Dx))
               + Dy.T.dot(Omega_y.dot(Dy)))

        # Normal denklem: (I + alpha * L_W) @ m_new = m_hat
        A = Identity + alpha * L_W            # (N x N) sparse

        # scipy sparse dogrusal sistem cozucu (Cholesky tabanli)
        m_new = spsolve(A, m_hat)

        # [0, 1] araligini koru
        m_new = np.clip(m_new, 0.0, 1.0)

        # IRLS yaklasim kriteri: cozum degisimi (lineer sistem reziduel degil)
        #   ||m_new - m_old|| / ||m_old||  < irls_tol => yakindi
        conv = np.linalg.norm(m_new - m) / (np.linalg.norm(m) + 1e-10)
        smooth_pct = np.linalg.norm(m_new - m_hat) / (np.linalg.norm(m_hat) + 1e-10) * 100
        print(f"    iter {it + 1:02d}/{n_iters}  |  conv={conv:.2e}  smoothing={smooth_pct:.1f}%")

        m = m_new
        if conv < irls_tol:
            print("    Erken yaklasim saglandi, durduruluyor.")
            break

    # H x W float32'ye geri sekillendirme
    M = m.reshape(H, W).astype(np.float32)
    return M


# ======================================================================
# REFLECTANCE HESABI
# ======================================================================

def compute_reflectance(rgb_01: np.ndarray,
                        illum_map: np.ndarray,
                        eps: float = 1e-6,
                        gamma: float = 0.8) -> np.ndarray:
    """
    Retinex: I = R * L  =>  R = I / L  (Lineer uzayda bolme)

    Parametreler
    ------------
    rgb_01    : (H, W, 3) float64, aralik [0, 1]  -- orijinal goruntu (RGB)
    illum_map : (H, W)    float32, aralik [0, 1]  -- IRLS illumination haritasi
    eps       : sifira bolmeyi engelleyen kucuk sayi
    gamma     : gamma duzeltme katsayisi (< 1 => aydinlatir, 1 => kapat)

    Dondutur
    --------
    R : np.ndarray, shape (H, W, 3), float32, aralik [0, 1]
        3 kanalli RGB reflectance haritasi.
    """
    # 1. Illumination haritasini (tek kanal) RGB boyutuna genislet (H, W, 3)
    illum_3d = np.repeat(illum_map[:, :, np.newaxis], 3, axis=2)

    # 2. Sifira bolmeyi engellemek icin eps ekleyerek dogrudan bol
    R = rgb_01 / (illum_3d + eps)

    # 3. Kirpma islemi (beyaz patlamalari onlemek icin)
    R = np.clip(R, 0.0, 1.0)

    # 4. Gamma duzeltmesi (cok karanlik kalirsa)
    R = np.power(R, gamma).astype(np.float32)

    return R


# ======================================================================
# DOSYA / KLASOR ISLEME
# ======================================================================

def collect_images(input_path: Path) -> list:
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTS:
            return [input_path]
        raise ValueError(f"Desteklenmeyen format: {input_path.suffix}")
    return sorted(
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def save_map(arr_01: np.ndarray, path: Path) -> None:
    """[0,1] float haritayi uint8 PNG olarak kaydet."""
    out = (arr_01 * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


def process(input_path: str,
            output_dir: str,
            alpha: float,
            n_iters: int,
            out_ext: str,
            save_reflectance: bool = True) -> None:

    inp = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    images = collect_images(inp)
    if not images:
        print("Hic goruntu bulunamadi.")
        return

    for img_path in images:
        print(f"\n[>] {img_path.name}")

        M = extract_illumination_map(str(img_path), alpha=alpha, n_iters=n_iters)

        illum_path = out / (img_path.stem + "-illumination" + out_ext)
        save_map(M, illum_path)
        print(f"  [OK] Illumination -> {illum_path.name}")

        if save_reflectance:
            bgr = cv2.imread(str(img_path))
            rgb_01 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            R_rgb = compute_reflectance(rgb_01, M)           # (H, W, 3) RGB float32
            R_bgr = cv2.cvtColor(R_rgb, cv2.COLOR_RGB2BGR)  # kayit icin BGR'ye cevir
            ref_path = out / (img_path.stem + "-reflectance" + out_ext)
            save_map(R_bgr, ref_path)
            print(f"  [OK] Reflectance  -> {ref_path.name}")

    print(f"\nToplam {len(images)} goruntu islendi.")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Retinex tabanli yapi-koruyucu aydinlatma haritasi cikarici.\n"
            "Optimizasyon: min_M ||M_hat-M||^2 + alpha*||W o nablaM||_1  (IRLS)"
        )
    )
    parser.add_argument("input",
                        help="Tek goruntu veya klasor yolu")
    parser.add_argument("--out", default="./output",
                        help="Cikti klasoru (varsayilan: ./output)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Duzgunlestirme katsayisi — kucuk=doku korur, buyuk=daha duzgun (varsayilan: 0.5)")
    parser.add_argument("--iters", type=int, default=10,
                        help="IRLS iterasyon sayisi (varsayilan: 10)")
    parser.add_argument("--ext", type=str, default=".png",
                        help="Cikti dosya uzantisi (varsayilan: .png)")
    parser.add_argument("--no-reflectance", action="store_true",
                        help="Reflectance haritasini kaydetme")
    args = parser.parse_args()

    process(
        input_path=args.input,
        output_dir=args.out,
        alpha=args.alpha,
        n_iters=args.iters,
        out_ext=args.ext,
        save_reflectance=not args.no_reflectance,
    )


if __name__ == "__main__":
    main()
