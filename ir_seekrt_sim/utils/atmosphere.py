"""
工具模块：大气透过率数据表
===========================
提供预计算的大气透过率查询表（基于 MODTRAN 典型场景），
以及插值查询接口。

数据来源：MODTRAN 标准大气（MLS - 中纬度夏季，农村气溶胶，
23km能见度），已针对中波(3-5μm)和长波(8-12μm)预计算。
"""

import numpy as np
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  预计算透过率查询表（MODTRAN 典型场景）
#  距离 [m]  vs  透过率 [-]
# ------------------------------------------------------------------ #

# 中波 3-5 μm（较好能见度，水汽适中）
MWIR_RANGE_M = np.array([
    0, 500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000, 50000
], dtype=float)

MWIR_TRANSMITTANCE = np.array([
    1.000, 0.920, 0.845, 0.712, 0.600, 0.430, 0.285, 0.220, 0.115, 0.060, 0.018, 0.002
], dtype=float)

# 长波 8-12 μm（较好能见度，CO2/H2O吸收更强）
LWIR_RANGE_M = np.array([
    0, 500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000, 50000
], dtype=float)

LWIR_TRANSMITTANCE = np.array([
    1.000, 0.880, 0.775, 0.600, 0.470, 0.290, 0.165, 0.120, 0.055, 0.025, 0.006, 0.001
], dtype=float)

# 插值对象（对数线性插值，更物理）
_mwir_interp = interp1d(
    MWIR_RANGE_M, np.log(MWIR_TRANSMITTANCE + 1e-10),
    kind="linear", fill_value="extrapolate"
)
_lwir_interp = interp1d(
    LWIR_RANGE_M, np.log(LWIR_TRANSMITTANCE + 1e-10),
    kind="linear", fill_value="extrapolate"
)


def lookup_transmittance(distance_m: float, band: str) -> float:
    """
    查询给定距离和波段的大气透过率（MODTRAN查找表插值）。

    Parameters
    ----------
    distance_m : 距离 [m]
    band       : "MWIR" 或 "LWIR"

    Returns
    -------
    tau : 大气透过率 (0~1)
    """
    band_up = band.upper()
    if band_up == "MWIR":
        log_tau = _mwir_interp(distance_m)
    elif band_up == "LWIR":
        log_tau = _lwir_interp(distance_m)
    else:
        logger.warning(f"未知波段 {band}，默认使用MWIR")
        log_tau = _mwir_interp(distance_m)

    tau = float(np.clip(np.exp(log_tau), 0.0, 1.0))
    logger.debug(f"查找表 τ({band},{distance_m:.0f}m) = {tau:.4f}")
    return tau


def get_transmittance_table(band: str) -> dict:
    """
    返回指定波段的完整透过率查找表。
    """
    if band.upper() == "MWIR":
        return {"range_m": MWIR_RANGE_M.tolist(), "transmittance": MWIR_TRANSMITTANCE.tolist()}
    else:
        return {"range_m": LWIR_RANGE_M.tolist(), "transmittance": LWIR_TRANSMITTANCE.tolist()}


def compute_band_average_transmittance(
    distance_m: float,
    lambda_min: float,
    lambda_max: float,
    n_samples: int = 100,
) -> float:
    """
    根据波段范围自动判断并返回透过率（简化版）。
    """
    lam_c_um = ((lambda_min + lambda_max) / 2) * 1e6

    if 3 <= lam_c_um <= 5:
        return lookup_transmittance(distance_m, "MWIR")
    elif 8 <= lam_c_um <= 12:
        return lookup_transmittance(distance_m, "LWIR")
    else:
        # 其他波段：简单指数衰减
        alpha = 5e-5
        return float(np.clip(np.exp(-alpha * distance_m), 0, 1))