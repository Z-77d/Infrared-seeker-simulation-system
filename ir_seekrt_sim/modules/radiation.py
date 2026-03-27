"""
模块一：辐射传输模型
======================
基于 Planck 黑体辐射公式，计算目标经大气传输后
到达相机入瞳的辐照度。

物理链：
    目标温度 T
        → 目标辐射亮度 L_bb(λ,T)  [Planck公式]
        → 目标出射亮度 L_target = ε · L_bb
        → 波段积分亮度 L_band [W·sr⁻¹·m⁻²]
        → 经大气衰减后到达入瞳辐照度 E_aperture [W·m⁻²]
"""

import numpy as np
from scipy import integrate
import logging

logger = logging.getLogger(__name__)

# 物理常数
h  = 6.62607015e-34   # Planck 常数 [J·s]
c  = 2.99792458e8     # 光速 [m/s]
kB = 1.380649e-23     # Boltzmann 常数 [J/K]


# ------------------------------------------------------------------ #
#  1. Planck 谱辐射亮度
# ------------------------------------------------------------------ #

def planck_spectral_radiance(wavelength: np.ndarray, temperature: float) -> np.ndarray:
    """
    计算黑体谱辐射亮度 L_bb(λ, T)。

    Parameters
    ----------
    wavelength  : 波长数组 [m]
    temperature : 温度 [K]

    Returns
    -------
    L_bb : 谱辐射亮度 [W·sr⁻¹·m⁻²·m⁻¹]
    """
    lam = np.asarray(wavelength, dtype=np.float64)
    T   = float(temperature)
    if T <= 0:
        raise ValueError(f"温度必须为正值，当前 T={T} K")

    c1 = 2.0 * h * c**2          # 第一辐射常数 [W·m²]
    c2 = h * c / kB              # 第二辐射常数 [m·K]

    exponent = c2 / (lam * T)
    # 防止溢出
    exponent = np.clip(exponent, None, 700)
    L_bb = c1 / (lam**5 * (np.exp(exponent) - 1.0))
    return L_bb


def planck_integrated_radiance(
    T: float,
    lam_min: float,
    lam_max: float,
    emissivity: float = 1.0,
    n_samples: int = 500,
) -> float:
    """
    波段积分辐射亮度（梯形积分）。

    Parameters
    ----------
    T          : 温度 [K]
    lam_min    : 最小波长 [m]
    lam_max    : 最大波长 [m]
    emissivity : 发射率 ε，默认 1.0（黑体）
    n_samples  : 积分采样点数

    Returns
    -------
    L_band : 波段积分亮度 [W·sr⁻¹·m⁻²]
    """
    wavelengths = np.linspace(lam_min, lam_max, n_samples)
    L_spec = planck_spectral_radiance(wavelengths, T)
    L_band = np.trapezoid(emissivity * L_spec, wavelengths) if hasattr(np, 'trapezoid') else np.trapz(emissivity * L_spec, wavelengths)
    logger.debug(f"L_band({T}K, {lam_min*1e6:.1f}-{lam_max*1e6:.1f}μm) = {L_band:.4e} W·sr⁻¹·m⁻²")
    return float(L_band)


# ------------------------------------------------------------------ #
#  2. 大气传输模型
# ------------------------------------------------------------------ #

def atmospheric_transmittance_exponential(
    distance: float,
    absorption_coeff: float,
    band: str = "MWIR",
) -> float:
    """
    简化指数大气传输模型（Beer-Lambert定律）。

        τ_atm = exp(-α · R)

    α 在中波/长波有典型值差异。

    Parameters
    ----------
    distance         : 传输距离 [m]
    absorption_coeff : 总衰减系数 α [1/m]，含吸收+散射
    band             : "MWIR" 或 "LWIR"

    Returns
    -------
    tau : 大气透过率 (0~1)
    """
    # 不同波段的典型修正因子（经验值）
    band_factor = {"MWIR": 1.0, "LWIR": 0.8}.get(band.upper(), 1.0)
    alpha = absorption_coeff * band_factor
    tau   = np.exp(-alpha * distance)
    tau   = float(np.clip(tau, 0.0, 1.0))
    logger.debug(f"大气透过率 τ={tau:.4f}（距离={distance:.0f}m, α={alpha:.2e}）")
    return tau


def atmospheric_transmittance_lowtran(
    distance: float,
    visibility_km: float = 23.0,
    humidity: float = 0.5,
    band: str = "MWIR",
) -> float:
    """
    LOWTRAN 简化近似模型（不依赖外部库）。
    基于 Kruse-Modified 能见度模型计算气溶胶散射。

    Parameters
    ----------
    distance       : 传输距离 [m]
    visibility_km  : 气象能见度 [km]
    humidity       : 相对湿度 (0~1)
    band           : 波段

    Returns
    -------
    tau : 大气透过率 (0~1)
    """
    V    = visibility_km
    dist_km = distance / 1000.0

    # Kruse 模型：散射系数 β = 3.912/V · (0.55/λ_ref)^q
    # 中波λ_ref=4μm，长波λ_ref=10μm
    if band.upper() == "MWIR":
        lam_ref = 4.0   # μm
    else:
        lam_ref = 10.0  # μm

    if V < 6:
        q = 0.585 * V**(1/3)
    elif V < 50:
        q = 1.3
    else:
        q = 1.6

    beta_vis  = 3.912 / V                              # 可见光参考波长散射系数 [1/km]
    beta_band = beta_vis * (0.55 / lam_ref)**q         # 目标波段散射系数

    # 水汽吸收（简化，MWIR 3.3μm 处有强吸收，均值化）
    water_abs = 0.02 * humidity * dist_km if band.upper() == "MWIR" else 0.005 * humidity * dist_km

    tau_aerosol = np.exp(-beta_band * dist_km)
    tau_water   = np.exp(-water_abs)
    tau         = float(np.clip(tau_aerosol * tau_water, 0.0, 1.0))
    logger.debug(f"LOWTRAN近似 τ={tau:.4f}（V={V}km, d={distance:.0f}m）")
    return tau


def get_atmospheric_transmittance(cfg: dict, distance: float, band: str) -> float:
    """
    根据配置选择大气传输模型。

    Parameters
    ----------
    cfg      : atmosphere 配置字典
    distance : 传输距离 [m]
    band     : 波段名称

    Returns
    -------
    tau : 大气透过率
    """
    model = cfg.get("model", "exponential").lower()

    if model == "custom":
        # 直接使用用户指定的透过率
        tau = float(cfg.get("transmittance", 0.75))

    elif model == "lowtran":
        tau = atmospheric_transmittance_lowtran(
            distance=distance,
            visibility_km=cfg.get("aerosol_visibility", 23.0),
            humidity=cfg.get("humidity", 0.5),
            band=band,
        )

    else:  # exponential (默认)
        tau = atmospheric_transmittance_exponential(
            distance=distance,
            absorption_coeff=cfg.get("absorption_coeff", 2e-5),
            band=band,
        )

    return tau


# ------------------------------------------------------------------ #
#  3. 目标与背景辐射收集计算
# ------------------------------------------------------------------ #

def compute_target_collected_power(
    L_band: float,
    tau_atm: float,
    target_area: float,
    distance: float,
    aperture_diameter: float,
    optics_transmission: float = 1.0,
) -> float:
    """
    计算目标进入光学系统的总辐射功率 P_target_W [W]。

        P = L_band * A_target * (A_aperture / R^2) * tau_atm * tau_opt

    Parameters
    ----------
    L_band            : 波段积分辐射亮度 [W·sr⁻¹·m⁻²]
    tau_atm           : 大气透过率
    target_area       : 目标面积 [m²]
    distance          : 目标距离 [m]
    aperture_diameter : 光学口径 [m]
    optics_transmission: 光学系统透过率

    Returns
    -------
    P_target_W : 目标总接收功率 [W]
    """
    A_aperture = np.pi * (aperture_diameter / 2.0)**2
    P = L_band * target_area * (A_aperture / distance**2) * tau_atm * optics_transmission
    logger.debug(f"目标捕获总功率 P = {P:.4e} W")
    return float(P)


def compute_background_focal_plane_irradiance(
    L_bg: float,
    tau_atm: float,
    f_number: float,
    optics_transmission: float = 1.0,
) -> float:
    """
    计算无穷远均匀背景在焦平面上产生的辐照度 E_fp [W/m²]。

        E_fp = (π / (4 * F#^2)) * L_bg * tau_atm * tau_opt

    Parameters
    ----------
    L_bg                : 背景波段积分辐射亮度 [W·sr⁻¹·m⁻²]
    tau_atm             : 大气透过率
    f_number            : 光学系统F数
    optics_transmission : 光学系统透过率

    Returns
    -------
    E_fp : 焦平面背景辐照度 [W/m²]
    """
    E_fp = (np.pi / (4.0 * f_number**2)) * L_bg * tau_atm * optics_transmission
    logger.debug(f"背景焦平面辐照度 E_fp = {E_fp:.4e} W/m²")
    return float(E_fp)


# ------------------------------------------------------------------ #
#  4. 目标与背景对比度
# ------------------------------------------------------------------ #

def compute_signal_contrast(
    target_temp: float,
    bg_temp: float,
    emissivity_target: float,
    emissivity_bg: float,
    lam_min: float,
    lam_max: float,
    n_samples: int = 500,
) -> dict:
    """
    计算目标相对背景的辐射亮度对比度。

    Returns
    -------
    dict: {L_target, L_bg, dL, contrast_ratio}
    """
    L_target = planck_integrated_radiance(target_temp, lam_min, lam_max, emissivity_target, n_samples)
    L_bg     = planck_integrated_radiance(bg_temp,     lam_min, lam_max, emissivity_bg,     n_samples)
    dL       = L_target - L_bg
    contrast = dL / (L_bg + 1e-30)

    result = {
        "L_target_Wsr1m2": L_target,
        "L_bg_Wsr1m2":     L_bg,
        "delta_L_Wsr1m2":  dL,
        "contrast_ratio":  contrast,
    }
    logger.info(f"辐射对比度: ΔL={dL:.4e} W·sr⁻¹·m⁻², 对比度={contrast:.4f}")
    return result


# ------------------------------------------------------------------ #
#  5. 完整辐射链计算（一次调用）
# ------------------------------------------------------------------ #

def compute_radiation_chain(target_cfg: dict, band_cfg: dict,
                            atm_cfg: dict, optics_cfg: dict) -> dict:
    """
    完整辐射传输链计算入口。

    Parameters
    ----------
    target_cfg  : 目标配置
    band_cfg    : 波段配置
    atm_cfg     : 大气配置
    optics_cfg  : 光学配置

    Returns
    -------
    dict: 各中间量和最终入瞳辐照度
    """
    T    = target_cfg["temperature"]
    eps  = target_cfg["emissivity"]
    A    = target_cfg["area"]
    R    = target_cfg["distance"]
    lmin = band_cfg["lambda_min"]
    lmax = band_cfg["lambda_max"]
    ns   = band_cfg.get("n_samples", 500)
    band_name = band_cfg.get("name", "MWIR")

    # 1. 波段积分亮度
    L_band = planck_integrated_radiance(T, lmin, lmax, eps, ns)

    # 2. 大气传输
    tau_atm = get_atmospheric_transmittance(atm_cfg, R, band_name)

    # 3. 目标总接收功率
    D    = optics_cfg["aperture_diameter"]
    t_op = optics_cfg.get("transmission", 0.85)
    P_tgt = compute_target_collected_power(L_band, tau_atm, A, R, D, t_op)

    result = {
        "L_band_Wsr1m2":   L_band,
        "tau_atm":          tau_atm,
        "P_target_W":       P_tgt,
        "target_temp_K":    T,
        "distance_m":       R,
        "band":             band_name,
    }
    return result