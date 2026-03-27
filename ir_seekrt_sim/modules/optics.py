"""
模块三：光学投影模型
====================
实现三维世界坐标 → 二维像素坐标的投影变换，
以及点扩散函数（PSF）生成和图像卷积。

功能：
1. 针孔相机模型（透视投影）
2. 相机内参矩阵构建
3. 世界坐标 → 像素坐标
4. PSF 模型（高斯 / Airy 盘）
5. 目标图像渲染（PSF 卷积）
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import j1
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

# 物理常数
h = 6.62607015e-34
c = 2.99792458e8


# ------------------------------------------------------------------ #
#  1. 相机内参矩阵
# ------------------------------------------------------------------ #

def build_intrinsic_matrix(
    focal_length: float,
    pixel_pitch: float,
    array_rows: int,
    array_cols: int,
) -> np.ndarray:
    """
    构建相机内参矩阵 K（3×3）。

        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

    fx = fy = f / pixel_pitch  （假设正方形像元）
    (cx, cy) = 阵列中心

    Parameters
    ----------
    focal_length : 焦距 [m]
    pixel_pitch  : 像元尺寸 [m]
    array_rows   : 阵列行数（高）
    array_cols   : 阵列列数（宽）

    Returns
    -------
    K : 3×3 内参矩阵（像素单位）
    """
    fx = focal_length / pixel_pitch
    fy = focal_length / pixel_pitch
    cx = (array_cols - 1) / 2.0
    cy = (array_rows - 1) / 2.0

    K = np.array([
        [fx,  0,  cx],
        [ 0, fy,  cy],
        [ 0,  0,   1],
    ], dtype=np.float64)

    logger.debug(f"内参矩阵: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    return K


# ------------------------------------------------------------------ #
#  2. 投影变换
# ------------------------------------------------------------------ #

def world_to_pixel(
    world_points: np.ndarray,
    K: np.ndarray,
    R_cam: Optional[np.ndarray] = None,
    t_cam: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    世界坐标 → 像素坐标（透视投影）。

        X_cam = R · X_world + t
        [u, v, 1]^T = (1/Z) · K · X_cam

    Parameters
    ----------
    world_points : (N, 3) 世界坐标 [m]
    K            : (3, 3) 内参矩阵
    R_cam        : (3, 3) 旋转矩阵（相机→世界），默认单位矩阵
    t_cam        : (3,)   平移向量，默认零向量

    Returns
    -------
    pixels : (N, 2) 像素坐标 [u(列), v(行)]，float
    """
    pts = np.atleast_2d(world_points).astype(np.float64)
    if R_cam is None:
        R_cam = np.eye(3)
    if t_cam is None:
        t_cam = np.zeros(3)

    # 转换到相机坐标系
    X_cam = (R_cam @ pts.T).T + t_cam  # (N, 3)

    # 过滤在相机后面的点
    Z = X_cam[:, 2]
    valid = Z > 0
    pixels = np.full((len(pts), 2), np.nan)

    if np.any(valid):
        Xv = X_cam[valid]
        # 归一化
        uv_h = (K @ Xv.T).T         # (N_valid, 3)
        z_v  = uv_h[:, 2:3]
        uv   = uv_h[:, :2] / z_v   # (N_valid, 2)
        pixels[valid] = uv

    return pixels   # [col, row] 即 [u, v]


def project_single_target(
    position_3d: List[float],
    K: np.ndarray,
    R_cam: Optional[np.ndarray] = None,
    t_cam: Optional[np.ndarray] = None,
    array_rows: int = 320,
    array_cols: int = 240,
) -> Tuple[Optional[int], Optional[int], bool]:
    """
    投影单个目标并判断是否在视场内。

    Returns
    -------
    (row, col, in_fov) : 整数像素坐标，in_fov=True 表示目标在图像范围内
    """
    pts = np.array([position_3d], dtype=np.float64)
    pixels = world_to_pixel(pts, K, R_cam, t_cam)
    u, v = pixels[0]

    col = int(round(u))
    row = int(round(v))
    in_fov = (0 <= col < array_cols) and (0 <= row < array_rows)

    if not in_fov:
        logger.warning(f"目标投影坐标 ({col}, {row}) 超出图像范围 ({array_cols}×{array_rows})")

    return row, col, in_fov


# ------------------------------------------------------------------ #
#  3. 点扩散函数 (PSF)
# ------------------------------------------------------------------ #

def gaussian_psf(sigma: float, size: int = None) -> np.ndarray:
    """
    生成二维高斯 PSF（归一化）。

    Parameters
    ----------
    sigma : 高斯 σ [像素]
    size  : 核大小（奇数），默认 6σ+1

    Returns
    -------
    psf : 归一化 PSF ndarray
    """
    if size is None:
        size = int(np.ceil(6 * sigma)) | 1  # 保证奇数
    if size % 2 == 0:
        size += 1

    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf


def airy_psf(
    aperture_diameter: float,
    focal_length: float,
    pixel_pitch: float,
    wavelength: float,
    size: int = None,
) -> np.ndarray:
    """
    生成 Airy 盘 PSF（衍射极限，归一化）。

        I(r) ∝ [2·J1(x)/x]²,  x = π·D·r/(λ·f)

    Parameters
    ----------
    aperture_diameter : 口径 D [m]
    focal_length      : 焦距 f [m]
    pixel_pitch       : 像元尺寸 [m]
    wavelength        : 参考波长 [m]
    size              : 核大小（像素，奇数）

    Returns
    -------
    psf : 归一化 Airy PSF
    """
    # Airy 盘第一零点半径（像素单位）
    r_airy_m    = 1.22 * wavelength * focal_length / aperture_diameter
    r_airy_pix  = r_airy_m / pixel_pitch

    if size is None:
        size = int(np.ceil(4 * r_airy_pix)) * 2 + 1
    if size % 2 == 0:
        size += 1

    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    r_pix = np.sqrt(x**2 + y**2)
    r_m   = r_pix * pixel_pitch

    # 避免 r=0 处除零
    x_arg  = np.pi * aperture_diameter * r_m / (wavelength * focal_length)
    x_safe = np.where(x_arg < 1e-10, 1e-10, x_arg)
    psf    = (2 * j1(x_safe) / x_safe) ** 2
    psf[r_pix == 0] = 1.0
    psf /= psf.sum()

    logger.debug(f"Airy PSF: r_airy={r_airy_pix:.2f} px, kernel={size}×{size}")
    return psf


def build_psf(optics_cfg: dict, band_cfg: dict, det_cfg: dict) -> np.ndarray:
    """
    根据配置构建 PSF。

    Returns
    -------
    psf : 归一化 PSF ndarray
    """
    model     = optics_cfg.get("psf_model", "gaussian").lower()
    lam_c     = (band_cfg["lambda_min"] + band_cfg["lambda_max"]) / 2.0
    px        = det_cfg["pixel_pitch"]
    D         = optics_cfg["aperture_diameter"]
    f         = optics_cfg["focal_length"]

    if model == "airy":
        psf = airy_psf(D, f, px, lam_c)
    else:  # gaussian (默认)
        sigma = optics_cfg.get("psf_sigma_pixels", 1.2)
        psf   = gaussian_psf(sigma)

    logger.debug(f"使用 {model.upper()} PSF，kernel shape={psf.shape}")
    return psf


# ------------------------------------------------------------------ #
#  4. 图像渲染
# ------------------------------------------------------------------ #

def render_point_target(
    electron_image: np.ndarray,
    target_row: int,
    target_col: int,
    target_electrons: float,
    psf: np.ndarray,
) -> np.ndarray:
    """
    在电子数图像上渲染点目标（PSF卷积方式）。

    流程：
    1. 在目标位置放置冲激函数（delta）
    2. 用 PSF 卷积扩散能量
    3. 叠加到背景图像

    Parameters
    ----------
    electron_image    : 背景+噪声的电子数图像 (rows, cols)
    target_row        : 目标行坐标
    target_col        : 目标列坐标
    target_electrons  : 目标总电子数 [e-]
    psf               : PSF 核

    Returns
    -------
    image_with_target : 含目标的电子数图像
    """
    rows, cols = electron_image.shape
    # 创建点目标图像
    point_img = np.zeros((rows, cols), dtype=np.float64)
    if 0 <= target_row < rows and 0 <= target_col < cols:
        point_img[target_row, target_col] = target_electrons

    # PSF 卷积
    target_spread = fftconvolve(point_img, psf, mode="same")

    result = electron_image + target_spread
    return result


def render_extended_target(
    electron_image: np.ndarray,
    target_center_row: int,
    target_center_col: int,
    target_electrons_total: float,
    target_size_pixels: Tuple[float, float],
    psf: np.ndarray,
) -> np.ndarray:
    """
    渲染面目标（高斯面源 + PSF卷积）。

    Parameters
    ----------
    target_size_pixels : (height_px, width_px) 目标在焦平面上的尺寸（像素）
    """
    rows, cols = electron_image.shape
    H, W = int(np.ceil(target_size_pixels[0])), int(np.ceil(target_size_pixels[1]))

    # 生成均匀面源
    source = np.zeros((rows, cols), dtype=np.float64)
    r0, c0 = target_center_row, target_center_col
    rmin   = max(0, r0 - H // 2)
    rmax   = min(rows, r0 + H // 2 + 1)
    cmin   = max(0, c0 - W // 2)
    cmax   = min(cols, c0 + W // 2 + 1)

    if rmax > rmin and cmax > cmin:
        n_pixels = (rmax - rmin) * (cmax - cmin)
        source[rmin:rmax, cmin:cmax] = target_electrons_total / n_pixels

    # PSF 卷积（考虑衍射和像差弥散）
    target_spread = fftconvolve(source, psf, mode="same")
    return electron_image + target_spread


def generate_ir_image(
    electron_image: np.ndarray,
    det_cfg: dict,
) -> np.ndarray:
    """
    将电子数图像量化为 DN 图像（最终红外图像）。

    Parameters
    ----------
    electron_image : 含噪声的电子数图像 [e-]
    det_cfg        : 探测器配置

    Returns
    -------
    dn_image : uint16 DN 图像
    """
    from modules.detector import electrons_to_dn
    fwc  = det_cfg["full_well_capacity"]
    bits = det_cfg["adc_bits"]
    dn   = electrons_to_dn(electron_image, fwc, bits)
    return dn


# ------------------------------------------------------------------ #
#  5. 目标尺寸在焦平面上的投影
# ------------------------------------------------------------------ #

def compute_target_size_pixels(
    target_physical_size_m: Tuple[float, float],
    distance_m: float,
    focal_length_m: float,
    pixel_pitch_m: float,
) -> Tuple[float, float]:
    """
    计算目标在焦平面上的投影尺寸（像素）。

        size_pixel = (L_physical / R) · (f / pixel_pitch)

    Parameters
    ----------
    target_physical_size_m : (height_m, width_m) 目标物理尺寸
    distance_m             : 目标距离 [m]
    focal_length_m         : 焦距 [m]
    pixel_pitch_m          : 像元尺寸 [m]

    Returns
    -------
    (height_px, width_px) : 像素尺寸（可为亚像素浮点数）
    """
    scale = focal_length_m / (distance_m * pixel_pitch_m)
    h_px  = target_physical_size_m[0] * scale
    w_px  = target_physical_size_m[1] * scale
    return float(h_px), float(w_px)