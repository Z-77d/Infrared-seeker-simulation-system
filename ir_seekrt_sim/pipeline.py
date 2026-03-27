"""
红外导引头仿真系统 - 主流程
============================
串联四个模块完成完整仿真：
  ① 辐射传输  →  ② 探测器响应  →  ③ 光学投影/成像  →  ④ 气动效应
"""

import sys
import os
import yaml
import logging
import numpy as np
import argparse
from pathlib import Path
from typing import Optional

# 确保 modules 可以被导入
sys.path.insert(0, os.path.dirname(__file__))

from modules.radiation   import compute_radiation_chain, compute_signal_contrast
from modules.detector    import NoiseModel, power_to_electrons, irradiance_to_electrons, electrons_to_dn, estimate_netd
from modules.optics      import (build_intrinsic_matrix, project_single_target,
                                  build_psf, render_point_target, render_extended_target,
                                  generate_ir_image, compute_target_size_pixels)
from modules.aerooptics  import apply_aero_optical_effects
from utils.image_io      import save_image, save_results_summary, normalize_to_8bit

# ------------------------------------------------------------------ #
#  日志配置
# ------------------------------------------------------------------ #

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  配置加载
# ------------------------------------------------------------------ #

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"已加载配置文件: {config_path}")
    return cfg


# ------------------------------------------------------------------ #
#  单帧仿真
# ------------------------------------------------------------------ #

# def simulate_frame(cfg: dict, rng: np.random.Generator, frame_idx: int = 0) -> dict:
#     """
#     执行单帧完整仿真，返回中间量和最终图像。
#
#     Parameters
#     ----------
#     cfg       : 完整配置字典
#     rng       : 随机数生成器
#     frame_idx : 帧序号（用于日志）
#
#     Returns
#     -------
#     results : 包含所有中间量和最终图像的字典
#     """
#     import copy
#     cfg = copy.deepcopy(cfg)
#     logger.info(f"===== 开始仿真 Frame {frame_idx} =====")
#     results = {"frame_idx": frame_idx}
#
#     target_cfg  = cfg["target"]
#
#     # ==== 引入导弹动态位置与运动模型 ====
#     dt = 0.05 # 时间步进，等效 20fps
#     time_sec = frame_idx * dt
#     # 假设导弹以快速向相机镜头斜向俯冲逼近！
#     v_x, v_y, v_z = 150.0, 50.0, -800.0
#     pos_3d = np.array(target_cfg.get("position_3d", [0.0, 0.0, target_cfg["distance"]]), dtype=np.float64)
#     pos_3d[0] += v_x * time_sec
#     pos_3d[1] += v_y * time_sec
#     pos_3d[2] += v_z * time_sec
#
#     current_distance = float(np.linalg.norm(pos_3d))
#     target_cfg["position_3d"] = pos_3d.tolist()
#     target_cfg["distance"] = current_distance
#     band_cfg    = cfg["band"]
#     atm_cfg     = cfg["atmosphere"]
#     optics_cfg  = cfg["optics"]
#     det_cfg     = cfg["detector"]
#     bg_cfg      = cfg.get("background", {})
#     aero_cfg    = cfg.get("aero_optics", {})
#
#     # --------------------------------------------------------
#     # 模块一：辐射传输计算
#     # --------------------------------------------------------
#     logger.info("【模块①】辐射传输计算...")
#     rad_result = compute_radiation_chain(target_cfg, band_cfg, atm_cfg, optics_cfg)
#     results.update({
#         "L_band_Wsr1m2": rad_result["L_band_Wsr1m2"],
#         "tau_atm":        rad_result["tau_atm"],
#         "P_target_W":     rad_result["P_target_W"],
#     })
#
#     # 背景辐射亮度（用于对比度计算）
#     bg_temp  = bg_cfg.get("temperature", 290.0)
#     bg_eps   = bg_cfg.get("emissivity", 0.95)
#     contrast = compute_signal_contrast(
#         target_cfg["temperature"], bg_temp,
#         target_cfg["emissivity"], bg_eps,
#         band_cfg["lambda_min"], band_cfg["lambda_max"],
#     )
#     results["contrast"] = contrast
#     logger.info(f"  辐射对比度: {contrast['contrast_ratio']:.4f}, "
#                 f"ΔL={contrast['delta_L_Wsr1m2']:.4e} W·sr⁻¹·m⁻²")
#     logger.info(f"  目标捕获总功率: P={rad_result['P_target_W']:.4e} W")
#
#     # --------------------------------------------------------
#     # 模块二：探测器响应（噪声模型）
#     # --------------------------------------------------------
#     logger.info("【模块②】探测器响应计算...")
#     noise_model = NoiseModel(det_cfg, rng=rng)
#
#     lam_c = (band_cfg["lambda_min"] + band_cfg["lambda_max"]) / 2.0
#     N_e_signal = power_to_electrons(
#         rad_result["P_target_W"],
#         det_cfg["integration_time"],
#         det_cfg["quantum_efficiency"],
#         lam_c,
#     )
#     results["N_e_signal"] = N_e_signal
#     logger.info(f"  目标捕获信号总电子数: N_e={N_e_signal:.4e} e-")
#
#     # 估算背景电子数
#     from modules.radiation import planck_integrated_radiance, compute_background_focal_plane_irradiance
#     L_bg = planck_integrated_radiance(bg_temp, band_cfg["lambda_min"],
#                                        band_cfg["lambda_max"], bg_eps)
#     E_bg_fp = compute_background_focal_plane_irradiance(
#         L_bg, rad_result["tau_atm"],
#         optics_cfg.get("f_number", 2.0),
#         optics_cfg.get("transmission", 0.85),
#     )
#     N_e_bg = irradiance_to_electrons(E_bg_fp, det_cfg["pixel_pitch"],
#                                       det_cfg["integration_time"],
#                                       det_cfg["quantum_efficiency"], lam_c)
#     results["N_e_background"] = N_e_bg
#     logger.info(f"  背景像元电子数: N_e_bg={N_e_bg:.4e} e-")
#
#     # NETD 估算
#     netd = estimate_netd(target_cfg, band_cfg, det_cfg, optics_cfg, N_e_signal)
#     results["NETD_K"] = netd
#     logger.info(f"  NETD 估算: {netd*1000:.2f} mK")
#
#     # --------------------------------------------------------
#     # 模块三：光学投影 + 图像生成
#     # --------------------------------------------------------
#     logger.info("【模块③】光学投影 + 图像生成...")
#
#     # 构建相机内参矩阵
#     K = build_intrinsic_matrix(
#         optics_cfg["focal_length"],
#         det_cfg["pixel_pitch"],
#         det_cfg["array_rows"],
#         det_cfg["array_cols"],
#     )
#     results["K_matrix"] = K
#
#     # 初始化背景电子数图像
#     rows = det_cfg["array_rows"]
#     cols = det_cfg["array_cols"]
#     bg_electron_image = np.full((rows, cols), N_e_bg, dtype=np.float64)
#     bg_with_noise     = noise_model.apply_noise(bg_electron_image)
#
#     # 设定外参 (R_cam, t_cam) - 随时间增加翻滚与姿态偏航，展示三维特征
#     theta_x = 0.5 + 0.2 * time_sec
#     theta_y = -0.4 - 0.5 * time_sec
#     theta_z = 2.0 * time_sec  # 滚转运动
#     from scipy.spatial.transform import Rotation
#     R_cam = Rotation.from_euler('xyz', [theta_x, theta_y, theta_z]).as_matrix()
#     t_cam = np.array(target_cfg["position_3d"], dtype=np.float64)
#
#     # 构建光学 PSF
#     psf = build_psf(optics_cfg, band_cfg, det_cfg)
#
#     # 渲染 3D 导弹目标及 PSF 卷积叠加
#     from modules.missile import render_3d_missile_target
#     electron_image = render_3d_missile_target(
#         bg_with_noise, K, R_cam, t_cam,
#         band_cfg, rad_result["tau_atm"], optics_cfg, det_cfg, psf
#     )
#     results["electron_image"] = electron_image
#
#     # 填补画图所需的老指标字段
#     from modules.optics import world_to_pixel
#     center_uv = world_to_pixel(np.array([[0,0,0]]), K, R_cam, t_cam)
#     if len(center_uv) > 0 and not np.isnan(center_uv[0][0]):
#         results["target_pixel_col"] = int(np.round(center_uv[0][0]))
#         results["target_pixel_row"] = int(np.round(center_uv[0][1]))
#         results["in_fov"] = True
#     else:
#         results["target_pixel_col"] = cols // 2
#         results["target_pixel_row"] = rows // 2
#         results["in_fov"] = False
#
#     # 量化为 DN 图像
#     dn_image = generate_ir_image(electron_image, det_cfg)
#     results["dn_image_before_aero"] = dn_image.copy()
#     logger.info(f"  DN图像: shape={dn_image.shape}, "
#                 f"min={dn_image.min()}, max={dn_image.max()}, "
#                 f"mean={dn_image.mean():.1f}")
#
#     # --------------------------------------------------------
#     # 模块四：气动光学效应
#     # --------------------------------------------------------
#     logger.info("【模块④】气动光学效应处理...")
#     final_image, aero_info = apply_aero_optical_effects(
#         dn_image, aero_cfg, optics_cfg, band_cfg, det_cfg, rng=rng
#     )
#     results["final_image"] = final_image
#     results["aero_info"]   = aero_info
#     logger.info(f"  最终图像: shape={final_image.shape}, "
#                 f"min={final_image.min()}, max={final_image.max()}, "
#                 f"mean={final_image.mean():.1f}")
#     logger.info(f"===== Frame {frame_idx} 仿真完成 =====\n")
#     return results
def simulate_frame(cfg: dict, rng: np.random.Generator, frame_idx: int = 0) -> dict:
    """
    执行单帧完整仿真，返回中间量和最终图像。

    Parameters
    ----------
    cfg       : 完整配置字典
    rng       : 随机数生成器
    frame_idx : 帧序号（用于日志）

    Returns
    -------
    results : 包含所有中间量和最终图像的字典
    """
    import copy
    cfg = copy.deepcopy(cfg)
    logger.info(f"===== 开始仿真 Frame {frame_idx} =====")
    results = {"frame_idx": frame_idx}

    target_cfg = cfg["target"]

    # ===================================================
    # ==== 引入导弹动态位置与运动模型 (尾追拦截场景) ====
    # ===================================================
    dt = 0.05  # 时间步进，等效 20fps
    time_sec = frame_idx * dt

    # 初始状态：目标在导引头前方 3500 米，向右偏 80 米，向上偏 40 米
    # # 相对速度：导引头正在高速逼近，接近速度为 600 m/s，并伴随 X/Y 方向的修正
    # initial_pos = np.array([80.0, -40.0, 3500.0], dtype=np.float64)
    # v_rel = np.array([-16.0, 8.0, -600.0])
    # 距离拉近到正前方 100 米，并且把偏移归零（否则在 100 米处偏 80 米就跑出镜头外了）
    initial_pos = np.array([0.0, 0.0, 100.0], dtype=np.float64)
    # 速度设为 0，让目标悬停，方便我们观察静态细节
    v_rel = np.array([0.0, 0.0, 0.0])

    pos_3d = initial_pos + v_rel * time_sec
    current_distance = float(np.linalg.norm(pos_3d))

    target_cfg["position_3d"] = pos_3d.tolist()
    target_cfg["distance"] = current_distance
    # ===================================================

    band_cfg = cfg["band"]
    atm_cfg = cfg["atmosphere"]
    optics_cfg = cfg["optics"]
    det_cfg = cfg["detector"]
    bg_cfg = cfg.get("background", {})
    aero_cfg = cfg.get("aero_optics", {})

    # --------------------------------------------------------
    # 模块一：辐射传输计算
    # --------------------------------------------------------
    logger.info("【模块①】辐射传输计算...")
    rad_result = compute_radiation_chain(target_cfg, band_cfg, atm_cfg, optics_cfg)
    results.update({
        "L_band_Wsr1m2": rad_result["L_band_Wsr1m2"],
        "tau_atm": rad_result["tau_atm"],
        "P_target_W": rad_result["P_target_W"],
    })

    # 背景辐射亮度（用于对比度计算）
    bg_temp = bg_cfg.get("temperature", 290.0)
    bg_eps = bg_cfg.get("emissivity", 0.95)
    contrast = compute_signal_contrast(
        target_cfg["temperature"], bg_temp,
        target_cfg["emissivity"], bg_eps,
        band_cfg["lambda_min"], band_cfg["lambda_max"],
    )
    results["contrast"] = contrast
    logger.info(f"  辐射对比度: {contrast['contrast_ratio']:.4f}, "
                f"ΔL={contrast['delta_L_Wsr1m2']:.4e} W·sr⁻¹·m⁻²")
    logger.info(f"  目标捕获总功率: P={rad_result['P_target_W']:.4e} W")

    # --------------------------------------------------------
    # 模块二：探测器响应（噪声模型）
    # --------------------------------------------------------
    logger.info("【模块②】探测器响应计算...")
    noise_model = NoiseModel(det_cfg, rng=rng)

    lam_c = (band_cfg["lambda_min"] + band_cfg["lambda_max"]) / 2.0
    N_e_signal = power_to_electrons(
        rad_result["P_target_W"],
        det_cfg["integration_time"],
        det_cfg["quantum_efficiency"],
        lam_c,
    )
    results["N_e_signal"] = N_e_signal
    logger.info(f"  目标捕获信号总电子数: N_e={N_e_signal:.4e} e-")

    # 估算背景电子数
    from modules.radiation import planck_integrated_radiance, compute_background_focal_plane_irradiance
    L_bg = planck_integrated_radiance(bg_temp, band_cfg["lambda_min"],
                                      band_cfg["lambda_max"], bg_eps)
    E_bg_fp = compute_background_focal_plane_irradiance(
        L_bg, rad_result["tau_atm"],
        optics_cfg.get("f_number", 2.0),
        optics_cfg.get("transmission", 0.85),
    )
    N_e_bg = irradiance_to_electrons(E_bg_fp, det_cfg["pixel_pitch"],
                                     det_cfg["integration_time"],
                                     det_cfg["quantum_efficiency"], lam_c)
    results["N_e_background"] = N_e_bg
    logger.info(f"  背景像元电子数: N_e_bg={N_e_bg:.4e} e-")

    # NETD 估算
    netd = estimate_netd(target_cfg, band_cfg, det_cfg, optics_cfg, N_e_signal)
    results["NETD_K"] = netd
    logger.info(f"  NETD 估算: {netd * 1000:.2f} mK")

    # --------------------------------------------------------
    # 模块三：光学投影 + 图像生成
    # --------------------------------------------------------
    logger.info("【模块③】光学投影 + 图像生成...")

    # 构建相机内参矩阵
    K = build_intrinsic_matrix(
        optics_cfg["focal_length"],
        det_cfg["pixel_pitch"],
        det_cfg["array_rows"],
        det_cfg["array_cols"],
    )
    results["K_matrix"] = K

    # 初始化背景电子数图像
    rows = det_cfg["array_rows"]
    cols = det_cfg["array_cols"]
    bg_electron_image = np.full((rows, cols), N_e_bg, dtype=np.float64)
    bg_with_noise = noise_model.apply_noise(bg_electron_image)

    # ===================================================
    # 设定外参 (R_cam, t_cam) - 展示尾追拦截的三维姿态特征
    # ===================================================
    from scipy.spatial.transform import Rotation

    # 引入偏航角（Yaw）、俯仰角（Pitch）和滚转角（Roll）
    yaw = 0.3 + 0.05 * np.sin(time_sec * 2)  # 基础偏航 0.3 rad (约17度) + 周期摆动
    pitch = -0.1  # 稍微低头
    roll = 2.0 * time_sec  # 导弹自身的高速滚转

    # 构建相对旋转矩阵
    R_cam = Rotation.from_euler('YXZ', [yaw, pitch, roll]).as_matrix()
    t_cam = np.array(target_cfg["position_3d"], dtype=np.float64)
    # ===================================================

    # 构建光学 PSF
    psf = build_psf(optics_cfg, band_cfg, det_cfg)

    # 渲染 3D 导弹目标及 PSF 卷积叠加
    from modules.missile import render_3d_missile_target
    electron_image = render_3d_missile_target(
        bg_with_noise, K, R_cam, t_cam,
        band_cfg, rad_result["tau_atm"], optics_cfg, det_cfg, psf
    )
    results["electron_image"] = electron_image

    # 填补画图所需的老指标字段
    from modules.optics import world_to_pixel
    center_uv = world_to_pixel(np.array([[0, 0, 0]]), K, R_cam, t_cam)
    if len(center_uv) > 0 and not np.isnan(center_uv[0][0]):
        results["target_pixel_col"] = int(np.round(center_uv[0][0]))
        results["target_pixel_row"] = int(np.round(center_uv[0][1]))
        results["in_fov"] = True
    else:
        results["target_pixel_col"] = cols // 2
        results["target_pixel_row"] = rows // 2
        results["in_fov"] = False

    # 量化为 DN 图像
    dn_image = generate_ir_image(electron_image, det_cfg)
    results["dn_image_before_aero"] = dn_image.copy()
    logger.info(f"  DN图像: shape={dn_image.shape}, "
                f"min={dn_image.min()}, max={dn_image.max()}, "
                f"mean={dn_image.mean():.1f}")

    # --------------------------------------------------------
    # 模块四：气动光学效应
    # --------------------------------------------------------
    logger.info("【模块④】气动光学效应处理...")
    final_image, aero_info = apply_aero_optical_effects(
        dn_image, aero_cfg, optics_cfg, band_cfg, det_cfg, rng=rng
    )
    results["final_image"] = final_image
    results["aero_info"] = aero_info
    logger.info(f"  最终图像: shape={final_image.shape}, "
                f"min={final_image.min()}, max={final_image.max()}, "
                f"mean={final_image.mean():.1f}")
    logger.info(f"===== Frame {frame_idx} 仿真完成 =====\n")
    return results


# ------------------------------------------------------------------ #
#  可视化（matplotlib）
# ------------------------------------------------------------------ #

def visualize_results(results: dict, cfg: dict, output_dir: str, frame_idx: int = 0):
    """生成并保存仿真结果对比图（4图布局）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # 解决中文字体显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ---- 1. 辐射传输：Planck曲线 ----
    ax1 = fig.add_subplot(gs[0, :2])
    from modules.radiation import planck_spectral_radiance
    lmin  = cfg["band"]["lambda_min"]
    lmax  = cfg["band"]["lambda_max"]
    lam_full = np.linspace(1e-6, 15e-6, 1000)
    T = cfg["target"]["temperature"]
    eps = cfg["target"]["emissivity"]
    L_spec = eps * planck_spectral_radiance(lam_full, T)
    ax1.semilogy(lam_full * 1e6, L_spec, "r-", linewidth=1.5, label=f"目标 T={T}K")
    ax1.axvspan(lmin * 1e6, lmax * 1e6, alpha=0.2, color="orange", label=f"仿真波段")
    bg_temp = cfg.get("background", {}).get("temperature", 290)
    L_bg_spec = planck_spectral_radiance(lam_full, bg_temp)
    ax1.semilogy(lam_full * 1e6, L_bg_spec, "b--", linewidth=1, label=f"背景 T={bg_temp}K")
    ax1.set_xlabel("波长 [μm]"); ax1.set_ylabel("谱辐射亮度 [W·sr⁻¹·m⁻²·m⁻¹]")
    ax1.set_title("① Planck 辐射曲线"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # ---- 2. 大气透过率 vs 距离 ----
    ax2 = fig.add_subplot(gs[0, 2:])
    from utils.atmosphere import lookup_transmittance
    band_name = cfg["band"].get("name", "MWIR")
    distances = np.linspace(0, 30000, 300)
    taus = [lookup_transmittance(d, band_name) for d in distances]
    ax2.plot(distances / 1000, taus, "g-", linewidth=2, label=band_name)
    ax2.axvline(cfg["target"]["distance"] / 1000, color="r", linestyle="--",
                label=f"目标距离 {cfg['target']['distance']/1000:.1f}km")
    ax2.axhline(results["tau_atm"], color="orange", linestyle=":", alpha=0.7,
                label=f"τ={results['tau_atm']:.3f}")
    ax2.set_xlabel("距离 [km]"); ax2.set_ylabel("大气透过率")
    ax2.set_title("② 大气透过率"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # ---- 3. 量化前 DN 图像 ----
    ax3 = fig.add_subplot(gs[1, :2])
    img_before = results.get("dn_image_before_aero", results["final_image"])
    im3 = ax3.imshow(img_before, cmap="inferno", aspect="auto")
    plt.colorbar(im3, ax=ax3, label="DN")
    if results["in_fov"]:
        ax3.plot(results["target_pixel_col"], results["target_pixel_row"],
                 "c+", markersize=10, markeredgewidth=2, label="目标")
        ax3.legend(fontsize=8)
    ax3.set_title("③ 仿真原始图像（无气动效应）")
    ax3.set_xlabel("列像素"); ax3.set_ylabel("行像素")

    # ---- 4. 气动效应后图像 ----
    ax4 = fig.add_subplot(gs[1, 2:])
    im4 = ax4.imshow(results["final_image"], cmap="inferno", aspect="auto")
    plt.colorbar(im4, ax=ax4, label="DN")
    ax4.set_title("④ 含气动效应的最终图像")
    ax4.set_xlabel("列像素"); ax4.set_ylabel("行像素")

    # ---- 5. 目标区域放大 ----
    ax5 = fig.add_subplot(gs[2, :2])
    r0, c0 = results["target_pixel_row"], results["target_pixel_col"]
    margin = 20
    rmin = max(0, r0 - margin); rmax = min(img_before.shape[0], r0 + margin)
    cmin = max(0, c0 - margin); cmax = min(img_before.shape[1], c0 + margin)
    crop_before = img_before[rmin:rmax, cmin:cmax]
    crop_after  = results["final_image"][rmin:rmax, cmin:cmax]
    im5 = ax5.imshow(crop_before, cmap="inferno", aspect="auto")
    plt.colorbar(im5, ax=ax5, label="DN")
    ax5.set_title(f"目标局部放大（气动前）±{margin}px")

    ax6 = fig.add_subplot(gs[2, 2:])
    im6 = ax6.imshow(crop_after, cmap="inferno", aspect="auto")
    plt.colorbar(im6, ax=ax6, label="DN")
    ax6.set_title(f"目标局部放大（气动后）±{margin}px")

    # 总标题 + 参数摘要
    P_tgt  = results.get("P_target_W", 0.0)
    N_e    = results["N_e_signal"]
    netd   = results["NETD_K"]
    tau    = results["tau_atm"]
    r0_aero = results.get("aero_info", {}).get("r0_cm", "N/A")
    r0_str  = f"{r0_aero:.2f} cm" if isinstance(r0_aero, float) else r0_aero

    fig.suptitle(
        f"红外导引头仿真结果 | Frame {frame_idx} | "
        f"T={T}K  R={cfg['target']['distance']}m  {band_name}\n"
        f"P_tgt={P_tgt:.3e}W  N_e={N_e:.3e}e-  "
        f"τ_atm={tau:.3f}  NETD={netd*1000:.1f}mK  r0={r0_str}",
        fontsize=11
    )

    fig_path = os.path.join(output_dir, f"simulation_result_frame{frame_idx:03d}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"结果图已保存: {fig_path}")
    return fig_path


# ------------------------------------------------------------------ #
#  主流程
# ------------------------------------------------------------------ #

def run_simulation(config_path: str, log_level: str = "INFO"):
    """
    完整仿真主流程。

    Parameters
    ----------
    config_path : 配置文件路径
    log_level   : 日志级别
    """
    setup_logging(log_level)
    cfg = load_config(config_path)

    # 随机数生成器
    seed = cfg.get("simulation", {}).get("random_seed", 42)
    rng  = np.random.default_rng(seed)
    logger.info(f"随机种子: {seed}")

    # 输出目录
    out_dir = cfg.get("simulation", {}).get("output_dir", "output")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"输出目录: {os.path.abspath(out_dir)}")

    n_frames = cfg.get("simulation", {}).get("n_frames", 1)
    fmt      = cfg.get("simulation", {}).get("output_format", "tiff")

    all_results = []

    for frame_idx in range(n_frames):
        results = simulate_frame(cfg, rng, frame_idx)
        all_results.append(results)

        # 保存最终图像
        img_path = os.path.join(out_dir, f"ir_image_frame{frame_idx:03d}")
        save_image(results["final_image"], img_path, fmt=fmt, normalize_for_preview=True)

        # 保存中间图像（若开启）
        if cfg.get("simulation", {}).get("save_intermediate", False):
            tmp = os.path.join(out_dir, f"ir_before_aero_frame{frame_idx:03d}")
            save_image(results["dn_image_before_aero"], tmp, fmt=fmt)

        # 可视化
        try:
            visualize_results(results, cfg, out_dir, frame_idx)
        except Exception as e:
            logger.warning(f"可视化失败（不影响仿真）: {e}")

    # 保存参数摘要（取第一帧）
    summary = {k: v for k, v in all_results[0].items()
               if not isinstance(v, np.ndarray)}
    summary["n_frames_completed"] = len(all_results)
    save_results_summary(summary, out_dir)

    # ===== 生成动态过程输出视频/GIF =====
    import glob
    from PIL import Image
    png_files = sorted(glob.glob(os.path.join(out_dir, "simulation_result_frame*.png")))
    if len(png_files) > 1:
        try:
            imgs = [Image.open(f) for f in png_files]
            gif_path = os.path.join(out_dir, "simulation_dynamic.gif")
            # 100 毫秒 = 10 FPS
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
            logger.info(f"🎥 动态全过程演示已生成: {os.path.abspath(gif_path)}")
        except Exception as e:
            logger.error(f"❌ 生成GIF动画失败: {e}")

    logger.info(f"\n✅ 仿真完成！共生成 {n_frames} 帧，结果保存在: {os.path.abspath(out_dir)}")
    return all_results


# ------------------------------------------------------------------ #
#  CLI 入口
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="红外导引头仿真系统")
    parser.add_argument(
        "--config", "-c",
        default="config/params.yaml",
        help="配置文件路径（默认: config/params.yaml）",
    )
    parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    args = parser.parse_args()

    results = run_simulation(args.config, args.log_level)