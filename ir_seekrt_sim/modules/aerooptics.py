"""
模块四：气动光学效应模型
=========================
模拟高速飞行器导引头受气动流场影响产生的图像退化：
1. 湍流相位屏（von Kármán 功率谱）→ 波前畸变 → PSF退化
2. 图像整体抖动（视轴随机偏移）
3. 热层/气动加热引起的折射率梯度模糊
4. 综合气动效应渲染

理论基础：
- Kolmogorov/von Kármán 大气湍流谱
- Fried 相干长度 r0 = 0.185 (λ²/Cn²·L)^(3/5)
- 相位屏傅里叶合成法
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import shift as ndimage_shift
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  1. 大气湍流参数计算
# ------------------------------------------------------------------ #

def compute_fried_parameter(
    Cn2: float,
    path_length: float,
    wavelength: float,
) -> float:
    """
    计算 Fried 相干长度 r0 [m]。

        r0 = 0.185 · (λ² / (Cn² · L))^(3/5)

    r0 越小，大气湍流越强，图像退化越严重。

    Parameters
    ----------
    Cn2         : 折射率结构常数 [m^(-2/3)]
    path_length : 传输路径长度 [m]
    wavelength  : 波长 [m]

    Returns
    -------
    r0 : Fried 相干长度 [m]
    """
    r0 = 0.185 * (wavelength**2 / (Cn2 * path_length))**(3/5)
    logger.debug(f"Fried参数 r0 = {r0*100:.2f} cm (Cn2={Cn2:.2e}, L={path_length:.0f}m)")
    return float(r0)


def compute_isoplanatic_angle(r0: float, path_length: float) -> float:
    """
    等晕角 θ0 [rad]，目标在此角度内波前畸变近似相同。

        θ0 ≈ 0.314 · (r0 / L)
    """
    theta0 = 0.314 * r0 / path_length
    return float(theta0)


# ------------------------------------------------------------------ #
#  2. 相位屏生成（von Kármán 谱）
# ------------------------------------------------------------------ #

def generate_phase_screen(
    N: int,
    D: float,
    r0: float,
    L0: float = 10.0,
    l0: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    使用傅里叶变换法生成 von Kármán 湍流相位屏。

    von Kármán 功率谱：
        PSD(f) = 0.023 · r0^(-5/3) · (f² + f0²)^(-11/6) · exp(-(f·l0)²)

    其中：
        f0 = 1/L0  （外尺度对应的空间频率）
        l0         （内尺度，Gaussian高频截止）

    Parameters
    ----------
    N   : 相位屏网格大小（像素）
    D   : 孔径物理尺寸 [m]（相位屏物理覆盖范围）
    r0  : Fried 相干长度 [m]
    L0  : 湍流外尺度 [m]
    l0  : 湍流内尺度 [m]
    rng : 随机数生成器

    Returns
    -------
    phase_screen : 相位畸变 [rad]，shape (N, N)
    """
    if rng is None:
        rng = np.random.default_rng()

    dx   = D / N                    # 空间采样间距 [m]
    df   = 1.0 / (N * dx)          # 频率采样间距 [1/m]
    f0   = 1.0 / L0                 # 外尺度截止频率

    # 构建频率网格
    freq_1d = np.fft.fftfreq(N, d=dx)
    fx, fy  = np.meshgrid(freq_1d, freq_1d)
    f2      = fx**2 + fy**2          # f² 网格

    # von Kármán 谱（避免 f=0 奇点）
    f2_safe = np.where (f2 == 0, 1e-30, f2)
    PSD = 0.023 * r0**(-5/3) * (f2_safe + f0**2)**(-11/6)
    # 内尺度高频截止（Gaussian）
    PSD *= np.exp(-(np.sqrt(f2_safe) * l0)**2)
    PSD[0, 0] = 0.0                  # 直流分量置零（无整体相位偏移）

    # 傅里叶合成：
    # 随机复高斯白噪声 → 乘以 √PSD → 逆变换
    amplitude = np.sqrt(PSD / (dx**2))
    noise_complex = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    noise_complex /= np.sqrt(2)

    phase_fft    = amplitude * noise_complex
    phase_screen = np.real(np.fft.ifft2(phase_fft)) * N**2 * df**2

    # 归一化到合理的 RMS
    # 理论 RMS 相位方差：σ² = 1.03·(D/r0)^(5/3) [rad²]
    sigma_theory = np.sqrt(1.03 * (D / r0)**(5/3))
    sigma_actual = np.std(phase_screen)
    if sigma_actual > 0:
        phase_screen = phase_screen / sigma_actual * sigma_theory

    logger.debug(f"相位屏生成: shape={phase_screen.shape}, σ={np.std(phase_screen):.3f} rad")
    return phase_screen


def phase_screen_to_psf(
    phase_screen: np.ndarray,
    pupil_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    将相位屏转换为系统 PSF（归一化强度）。

    PSF = |FT{ P · exp(i·φ) }|²

    Parameters
    ----------
    phase_screen : 相位畸变 [rad], shape (N, N)
    pupil_mask   : 圆形光瞳掩模（None则自动生成）

    Returns
    -------
    psf : 归一化 PSF（峰值=1）
    """
    N = phase_screen.shape[0]

    if pupil_mask is None:
        # 生成圆形光瞳掩模
        half = N // 2
        y, x = np.mgrid[-half:half, -half:half]
        if N % 2 == 0:
            y = y + 0.5; x = x + 0.5
        r = np.sqrt(x**2 + y**2)
        pupil_mask = (r <= half * 0.9).astype(float)

    # 复振幅
    pupil = pupil_mask * np.exp(1j * phase_screen)

    # 傅里叶变换到焦面
    # 中心化：先 fftshift 使零频在中心
    psf_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    psf       = np.abs(psf_field)**2

    # 归一化
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf = psf / psf_sum

    return psf


# ------------------------------------------------------------------ #
#  3. 气动效应图像处理
# ------------------------------------------------------------------ #

def apply_turbulence_blur(
    image: np.ndarray,
    aero_psf: np.ndarray,
) -> np.ndarray:
    """
    用气动湍流 PSF 对图像做卷积（图像模糊）。

    Parameters
    ----------
    image    : 输入图像（float或uint16）
    aero_psf : 气动 PSF（归一化）

    Returns
    -------
    blurred : 模糊后图像（与输入同类型）
    """
    dtype_in = image.dtype
    img_f    = image.astype(np.float64)

    # 截取PSF中心区域（避免太大的核降低效率）
    psf = _crop_psf(aero_psf, max_size=min(image.shape) // 4 * 2 + 1)

    blurred = fftconvolve(img_f, psf, mode="same")
    blurred = np.clip(blurred, 0, None)

    # 恢复原始类型
    if np.issubdtype(dtype_in, np.integer):
        max_val = np.iinfo(dtype_in).max
        blurred = np.clip(np.round(blurred), 0, max_val).astype(dtype_in)
    else:
        blurred = blurred.astype(dtype_in)

    return blurred


def _crop_psf(psf: np.ndarray, max_size: int = 63) -> np.ndarray:
    """裁剪PSF到合理大小并重新归一化。"""
    H, W  = psf.shape
    h_out = min(H, max_size) | 1   # 保证奇数
    w_out = min(W, max_size) | 1

    r0, c0 = np.unravel_index(np.argmax(psf), psf.shape)
    r_min  = max(0, r0 - h_out // 2)
    r_max  = r_min + h_out
    c_min  = max(0, c0 - w_out // 2)
    c_max  = c_min + w_out

    cropped = psf[r_min:r_max, c_min:c_max].copy()
    s = cropped.sum()
    if s > 0:
        cropped /= s
    return cropped


def apply_image_jitter(
    image: np.ndarray,
    jitter_sigma_pixels: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    对图像施加整体随机抖动（模拟气流引起的视轴抖动）。

    使用双三次插值（scipy实现亚像素平移）。

    Parameters
    ----------
    image                : 输入图像
    jitter_sigma_pixels  : 抖动幅度 σ [像素]
    rng                  : 随机数生成器

    Returns
    -------
    jittered_image : 抖动后图像
    (dr, dc)       : 本次抖动量 [像素]
    """
    if rng is None:
        rng = np.random.default_rng()

    dr = float(rng.normal(0, jitter_sigma_pixels))
    dc = float(rng.normal(0, jitter_sigma_pixels))

    dtype_in = image.dtype
    img_f    = image.astype(np.float64)

    # scipy ndimage.shift: (row, col) 方向的平移
    jittered = ndimage_shift(img_f, shift=(dr, dc), mode="reflect", order=3)
    jittered = np.clip(jittered, 0, None)

    if np.issubdtype(dtype_in, np.integer):
        max_val  = np.iinfo(dtype_in).max
        jittered = np.clip(np.round(jittered), 0, max_val).astype(dtype_in)
    else:
        jittered = jittered.astype(dtype_in)

    logger.debug(f"图像抖动: Δrow={dr:.3f}px, Δcol={dc:.3f}px")
    return jittered, (dr, dc)


def apply_aero_heating_haze(
    image: np.ndarray,
    flow_velocity: float,
    path_length: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    气动加热引起的热层折射率梯度模糊（各向异性模糊）。

    高速流场中，激波/边界层产生折射率梯度，
    等效为沿飞行方向的横向模糊。用各向异性高斯核近似。

    Parameters
    ----------
    flow_velocity : 飞行速度 [m/s]（马赫数近似: M ≈ v/340）
    path_length   : 气动光学路径长度 [m]

    Returns
    -------
    haze_image : 处理后图像
    """
    if rng is None:
        rng = np.random.default_rng()

    # 马赫数
    M = flow_velocity / 340.0

    # 气动光学效应强度与马赫数和路径长度正相关（经验公式）
    if M < 1.0:
        # 亚音速：效果微弱
        sigma_x = 0.2 * M
        sigma_y = 0.1 * M
    elif M < 3.0:
        # 跨音速/低超音速
        sigma_x = 0.5 * (M - 0.5) * np.sqrt(path_length / 100)
        sigma_y = 0.2 * (M - 0.5) * np.sqrt(path_length / 100)
    else:
        # 高超音速（非常强的热层效应）
        sigma_x = 1.5 * np.log(M) * np.sqrt(path_length / 100)
        sigma_y = 0.5 * np.log(M) * np.sqrt(path_length / 100)

    sigma_x = max(sigma_x, 0.1)
    sigma_y = max(sigma_y, 0.1)

    # 生成各向异性高斯核
    size = int(np.ceil(max(sigma_x, sigma_y) * 6)) | 1
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    kernel = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
    kernel /= kernel.sum()

    dtype_in = image.dtype
    img_f    = image.astype(np.float64)
    result   = fftconvolve(img_f, kernel, mode="same")
    result   = np.clip(result, 0, None)

    if np.issubdtype(dtype_in, np.integer):
        max_val = np.iinfo(dtype_in).max
        result  = np.clip(np.round(result), 0, max_val).astype(dtype_in)
    else:
        result  = result.astype(dtype_in)

    logger.debug(f"气动热层模糊: M={M:.2f}, σx={sigma_x:.3f}px, σy={sigma_y:.3f}px")
    return result


# ------------------------------------------------------------------ #
#  4. 多层相位屏（更精确的湍流仿真）
# ------------------------------------------------------------------ #

def generate_multi_layer_psf(
    N: int,
    D: float,
    r0: float,
    n_screens: int = 3,
    L0: float = 10.0,
    l0: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    使用多层相位屏合成更真实的大气 PSF。

    每层独立生成相位屏，复振幅相乘后统一做 FFT。

    Parameters
    ----------
    n_screens : 相位屏层数（越多越精确，但越慢）

    Returns
    -------
    psf : 归一化复合 PSF
    """
    if rng is None:
        rng = np.random.default_rng()

    # 每层分配 r0（等效：r0_layer^(-5/3) · n = r0_total^(-5/3)）
    r0_layer = r0 * n_screens**(3/5)

    # 圆形光瞳掩模
    half = N // 2
    y, x = np.mgrid[-half:half, -half:half]
    r    = np.sqrt(x**2 + y**2)
    pupil_mask = (r <= half * 0.9).astype(float)

    # 累积相位
    total_phase = np.zeros((N, N), dtype=np.float64)
    for i in range(n_screens):
        screen = generate_phase_screen(N, D, r0_layer, L0=L0, l0=l0, rng=rng)
        total_phase += screen

    psf = phase_screen_to_psf(total_phase, pupil_mask)
    return psf


# ------------------------------------------------------------------ #
#  5. 完整气动效应处理（主入口）
# ------------------------------------------------------------------ #

def apply_aero_optical_effects(
    image: np.ndarray,
    aero_cfg: dict,
    optics_cfg: dict,
    band_cfg: dict,
    det_cfg: dict,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, dict]:
    """
    对输入图像施加完整气动光学效应。

    效果叠加顺序：
    1. 气动加热热层模糊（各向异性）
    2. 大气湍流 PSF 退化（相位屏）
    3. 图像整体抖动

    Parameters
    ----------
    image     : 输入红外图像（uint16 DN值）
    aero_cfg  : 气动光学配置
    optics_cfg: 光学配置
    band_cfg  : 波段配置
    det_cfg   : 探测器配置
    rng       : 随机数生成器

    Returns
    -------
    output_image : 含气动效应的图像
    info_dict    : 各中间参数
    """
    if rng is None:
        rng = np.random.default_rng()

    if not aero_cfg.get("enable", True):
        logger.info("气动效应已禁用，跳过处理")
        return image.copy(), {}

    Cn2        = aero_cfg.get("Cn2", 1e-14)
    path_L     = aero_cfg.get("path_length", 100.0)
    L0         = aero_cfg.get("L0", 10.0)
    l0         = aero_cfg.get("l0", 0.01)
    jitter_sig = aero_cfg.get("jitter_sigma_pixels", 0.5)
    n_screens  = aero_cfg.get("n_phase_screens", 3)
    flow_v     = aero_cfg.get("flow_velocity", 300.0)

    lam_c = (band_cfg["lambda_min"] + band_cfg["lambda_max"]) / 2.0
    D     = optics_cfg["aperture_diameter"]
    px    = det_cfg["pixel_pitch"]
    rows  = det_cfg["array_rows"]
    cols  = det_cfg["array_cols"]

    # 相位屏尺寸：取探测器阵列较小边
    N_screen = min(rows, cols)
    # 取偶数（FFT效率）
    N_screen = N_screen if N_screen % 2 == 0 else N_screen - 1

    current_image = image.copy()
    info = {}

    # Step 1: 气动加热热层模糊
    logger.info("Step 1: 气动加热热层模糊...")
    current_image = apply_aero_heating_haze(current_image, flow_v, path_L, rng)

    # Step 2: 湍流 PSF 退化
    logger.info(f"Step 2: 湍流相位屏 PSF（{n_screens}层）...")
    r0 = compute_fried_parameter(Cn2, path_L, lam_c)
    info["r0_m"] = r0
    info["r0_cm"] = r0 * 100

    # 物理光瞳尺寸（相位屏空间覆盖范围 = 口径 D）
    aero_psf = generate_multi_layer_psf(
        N=N_screen, D=D, r0=r0,
        n_screens=n_screens, L0=L0, l0=l0, rng=rng
    )
    info["aero_psf"] = aero_psf

    # PSF 需要缩放到像素域（相位屏中心区域对应焦平面弥散）
    # 从孔径域 PSF 换算像素弥散尺寸：σ_px = λ·f/(r0·px)
    sigma_turb_px = lam_c * optics_cfg["focal_length"] / (r0 * px)
    info["sigma_turbulence_px"] = sigma_turb_px

    # 用高斯近似（更稳定）替代直接使用 N×N 相位屏 PSF
    from modules.optics import gaussian_psf
    turb_psf_px = gaussian_psf(sigma=sigma_turb_px)
    current_image = apply_turbulence_blur(current_image, turb_psf_px)

    # Step 3: 图像抖动
    logger.info(f"Step 3: 图像抖动（σ={jitter_sig:.2f}px）...")
    current_image, (dr, dc) = apply_image_jitter(current_image, jitter_sig, rng)
    info["jitter_dr_px"] = dr
    info["jitter_dc_px"] = dc

    logger.info("气动光学效应处理完成")
    return current_image, info