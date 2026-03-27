"""
模块二：红外探测器响应模型
===========================
将入瞳辐照度转换为像元数字量（DN值），
包含完整的噪声链模型。

物理链：
    入瞳辐照度 E [W/m²]
        → 像元接收功率 P_pixel [W]
        → 光生电荷数 N_e [e-]
        → 含各类噪声后总电荷 N_total [e-]
        → ADC 量化 → DN 值 [0 ~ 2^bits-1]
"""

import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# 物理常数
h  = 6.62607015e-34
c  = 2.99792458e8
kB = 1.380649e-23
q_e = 1.602176634e-19   # 电子电荷 [C]


# ------------------------------------------------------------------ #
#  1. 光子 → 电荷 转换
# ------------------------------------------------------------------ #

def power_to_electrons(
    P_watt: float,
    integration_time: float,
    quantum_efficiency: float,
    lam_mean: float,
) -> float:
    """
    将接收到的辐射功率转换为光生电子数。

        N_ph = P · t_int / (hν) = P · t_int · λ / (h·c)
        N_e  = η · N_ph

    Parameters
    ----------
    P_watt            : 接收功率 [W]
    integration_time  : 积分时间 [s]
    quantum_efficiency: 量子效率 η (0~1)
    lam_mean          : 波段中心波长 [m]

    Returns
    -------
    N_e : 光生电子数 [e-]
    """
    E_photon = h * c / lam_mean
    N_ph = P_watt * integration_time / E_photon
    N_e  = quantum_efficiency * N_ph
    logger.debug(f"P={P_watt:.4e}W, N_ph={N_ph:.4e}, N_e={N_e:.4e}")
    return float(N_e)


def irradiance_to_electrons(
    E_focal_plane: float,
    pixel_pitch: float,
    integration_time: float,
    quantum_efficiency: float,
    lam_mean: float,
) -> float:
    """
    将焦平面辐照度转换为单像元光生电子数。

    Parameters
    ----------
    E_focal_plane     : 焦平面辐照度 [W/m²]
    pixel_pitch       : 像元尺寸 [m]
    integration_time  : 积分时间 [s]
    quantum_efficiency: 量子效率 η (0~1)
    lam_mean          : 波段中心波长 [m]
    """
    A_pixel = pixel_pitch ** 2
    P_pixel = E_focal_plane * A_pixel                        # 像元接收功率 [W]
    return power_to_electrons(P_pixel, integration_time, quantum_efficiency, lam_mean)


# ------------------------------------------------------------------ #
#  2. 噪声模型
# ------------------------------------------------------------------ #

class NoiseModel:
    """
    完整噪声链模型：
    - 散粒噪声（Poisson → Gaussian近似）
    - 读出噪声（高斯）
    - 暗电流噪声
    - 固定模式噪声（FPN）
    """

    def __init__(self, det_cfg: dict, rng: Optional[np.random.Generator] = None):
        self.read_noise  = det_cfg.get("read_noise_electrons", 50.0)
        self.dark_curr   = det_cfg.get("dark_current", 1e4)           # [e-/s]
        self.fwc         = det_cfg.get("full_well_capacity", 2e6)     # [e-]
        self.fpn_sigma   = det_cfg.get("fpn_sigma", 0.01)            # 相对 σ
        self.t_int       = det_cfg.get("integration_time", 10e-3)    # [s]
        self.rows        = det_cfg.get("array_rows", 320)
        self.cols        = det_cfg.get("array_cols", 240)
        self.rng         = rng if rng is not None else np.random.default_rng()

        # 预生成固定模式噪声矩阵（器件固有，每次仿真固定）
        self._fpn_map = self.rng.normal(1.0, self.fpn_sigma, (self.rows, self.cols))
        self._fpn_map = np.clip(self._fpn_map, 0.5, 2.0)

    def apply_noise(self, signal_electrons: np.ndarray) -> np.ndarray:
        """
        对电子数图像添加所有噪声。

        Parameters
        ----------
        signal_electrons : 信号电子数矩阵 [e-]，shape (rows, cols)

        Returns
        -------
        total_electrons : 含噪声的总电子数矩阵 [e-]
        """
        N_e = signal_electrons.copy()

        # 1. 固定模式噪声（像元响应不均匀性）
        N_e = N_e * self._fpn_map[:N_e.shape[0], :N_e.shape[1]]

        # 2. 暗电流
        N_dark = self.dark_curr * self.t_int  # [e-]

        # 3. 暗电流散粒噪声（Poisson）
        dark_shot_sigma = np.sqrt(N_dark)
        N_dark_noisy = N_dark + self.rng.normal(0.0, dark_shot_sigma, N_e.shape)

        # 4. 光子散粒噪声（Poisson，对大数用Gaussian近似）
        shot_sigma = np.sqrt(np.maximum(N_e, 0.0))
        N_e_noisy  = N_e + self.rng.normal(0.0, shot_sigma, N_e.shape)

        # 5. 读出噪声（高斯）
        N_read = self.rng.normal(0.0, self.read_noise, N_e.shape)

        # 汇总
        N_total = N_e_noisy + N_dark_noisy + N_read

        # 满阱截断（clip到 FWC）
        N_total = np.clip(N_total, 0.0, self.fwc)

        return N_total

    def compute_noise_sigma(self, N_e: float) -> float:
        """
        计算给定信号电子数时的总噪声 σ [e-]（标量，用于NETD估算）。
        """
        sigma_shot = np.sqrt(max(N_e, 0))
        sigma_dark = np.sqrt(self.dark_curr * self.t_int)
        sigma_total = np.sqrt(sigma_shot**2 + sigma_dark**2 + self.read_noise**2)
        return float(sigma_total)


# ------------------------------------------------------------------ #
#  3. ADC 量化
# ------------------------------------------------------------------ #

def electrons_to_dn(
    electrons: np.ndarray,
    full_well_capacity: float,
    adc_bits: int,
) -> np.ndarray:
    """
    将电子数转换为数字量 DN。

        DN = round( N_e / FWC * (2^bits - 1) )

    Parameters
    ----------
    electrons          : 电子数矩阵
    full_well_capacity : 满阱容量 [e-]
    adc_bits           : ADC 位深

    Returns
    -------
    dn : uint16 DN 值矩阵
    """
    max_dn = 2**adc_bits - 1
    dn_float = electrons / full_well_capacity * max_dn
    dn = np.round(dn_float).astype(np.int32)
    dn = np.clip(dn, 0, max_dn)
    return dn.astype(np.uint16)


# ------------------------------------------------------------------ #
#  4. NETD 估算
# ------------------------------------------------------------------ #

def estimate_netd(
    target_cfg: dict,
    band_cfg: dict,
    det_cfg: dict,
    optics_cfg: dict,
    N_e_signal: float,
    delta_T: float = 1.0,
) -> float:
    """
    估算噪声等效温差 NETD [K]。

        NETD = σ_noise / (dN_e/dT)

    Parameters
    ----------
    N_e_signal : 信号电子数
    delta_T    : 温度微分步长 [K]

    Returns
    -------
    netd : NETD [K]
    """
    from modules.radiation import planck_integrated_radiance

    T     = target_cfg["temperature"]
    lmin  = band_cfg["lambda_min"]
    lmax  = band_cfg["lambda_max"]
    eps   = target_cfg["emissivity"]
    A     = target_cfg["area"]
    R     = target_cfg["distance"]
    D     = optics_cfg["aperture_diameter"]
    t_op  = optics_cfg.get("transmission", 0.85)
    px    = det_cfg["pixel_pitch"]
    t_int = det_cfg["integration_time"]
    qe    = det_cfg["quantum_efficiency"]
    lam_c = (lmin + lmax) / 2.0

    # dL/dT 数值微分
    L1 = planck_integrated_radiance(T - delta_T/2, lmin, lmax, eps)
    L2 = planck_integrated_radiance(T + delta_T/2, lmin, lmax, eps)
    dL_dT = (L2 - L1) / delta_T

    # dP/dT (目标全孔径捕获功率对温度微分)
    A_ap = np.pi * (D/2)**2
    dP_dT = dL_dT * A * (A_ap / R**2) * t_op

    # dN_e/dT
    E_ph = 6.62607015e-34 * 2.99792458e8 / lam_c
    dNe_dT = qe * t_int * dP_dT / E_ph

    # 噪声模型估算 σ
    noise_model = NoiseModel(det_cfg)
    sigma_noise = noise_model.compute_noise_sigma(N_e_signal)

    netd = sigma_noise / (abs(dNe_dT) + 1e-30)
    logger.info(f"NETD 估算 = {netd*1000:.2f} mK  (σ={sigma_noise:.1f}e-, dNe/dT={dNe_dT:.4e})")
    return float(netd)


# ------------------------------------------------------------------ #
#  5. 完整探测器响应（一次调用）
# ------------------------------------------------------------------ #

def compute_detector_response(
    P_target_W: float,
    target_pixel_pos: Tuple[int, int],
    det_cfg: dict,
    band_cfg: dict,
    noise_model: NoiseModel,
    background_electrons: Optional[float] = None,
) -> np.ndarray:
    """
    生成包含信号和背景的完整探测器电子数图像（未量化）。

    Parameters
    ----------
    P_target_W          : 目标进入光学系统的总辐射功率 [W]
    target_pixel_pos    : 目标中心像素坐标 (row, col)
    det_cfg             : 探测器配置
    band_cfg            : 波段配置
    noise_model         : NoiseModel 实例
    background_electrons: 背景像元电子数（若None则自动估算）

    Returns
    -------
    electron_image : 电子数图像 ndarray [e-], shape (rows, cols)
    """
    rows  = det_cfg["array_rows"]
    cols  = det_cfg["array_cols"]
    px    = det_cfg["pixel_pitch"]
    t_int = det_cfg["integration_time"]
    qe    = det_cfg["quantum_efficiency"]
    lam_c = (band_cfg["lambda_min"] + band_cfg["lambda_max"]) / 2.0

    # 目标信号电子数
    N_e_target = power_to_electrons(P_target_W, t_int, qe, lam_c)

    # 背景电子数（简化：背景辐照度约为目标的某个比例）
    if background_electrons is None:
        background_electrons = N_e_target * 0.05  # 简化：背景约为目标5%

    # 初始化全图为背景
    electron_image = np.full((rows, cols), background_electrons, dtype=np.float64)

    # 在目标位置叠加目标信号
    r0, c0 = target_pixel_pos
    if 0 <= r0 < rows and 0 <= c0 < cols:
        electron_image[r0, c0] += N_e_target

    # 添加噪声
    electron_image = noise_model.apply_noise(electron_image)

    return electron_image