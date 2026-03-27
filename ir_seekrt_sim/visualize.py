"""
独立可视化工具
==============
提供丰富的仿真结果可视化函数，可单独调用。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import logging

logger = logging.getLogger(__name__)


def plot_planck_curves(temperatures: list, lam_range=(1e-6, 15e-6),
                       bands=None, output_path: str = None):
    """
    绘制多个温度的 Planck 辐射曲线。

    Parameters
    ----------
    temperatures : 温度列表 [K]
    lam_range    : 波长范围 (min, max) [m]
    bands        : 波段高亮区域列表，每项 (lam_min, lam_max, label)
    output_path  : 保存路径
    """
    from modules.radiation import planck_spectral_radiance

    fig, ax = plt.subplots(figsize=(10, 6))
    lam = np.linspace(lam_range[0], lam_range[1], 2000)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(temperatures)))
    for T, color in zip(temperatures, colors):
        L = planck_spectral_radiance(lam, T)
        ax.semilogy(lam * 1e6, L, color=color, linewidth=1.8, label=f"{T} K")

    if bands:
        band_colors = ["orange", "cyan", "lime"]
        for i, (l1, l2, label) in enumerate(bands):
            ax.axvspan(l1 * 1e6, l2 * 1e6, alpha=0.15,
                       color=band_colors[i % len(band_colors)], label=label)

    ax.set_xlabel("波长 [μm]", fontsize=12)
    ax.set_ylabel("谱辐射亮度 [W·sr⁻¹·m⁻²·m⁻¹]", fontsize=12)
    ax.set_title("Planck 黑体辐射曲线", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim([lam_range[0] * 1e6, lam_range[1] * 1e6])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Planck曲线已保存: {output_path}")
    plt.close()


def plot_psf_comparison(psf_gaussian, psf_airy=None, psf_turbulent=None,
                        output_path: str = None):
    """绘制不同 PSF 的对比图。"""
    psfs  = [psf_gaussian]
    names = ["高斯 PSF"]
    if psf_airy is not None:
        psfs.append(psf_airy); names.append("Airy 盘 PSF")
    if psf_turbulent is not None:
        psfs.append(psf_turbulent); names.append("气动湍流 PSF")

    n = len(psfs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, psf, name in zip(axes, names, names):
        im = ax.imshow(psf, cmap="hot", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("列"); ax.set_ylabel("行")

    fig.suptitle("PSF 对比", fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_noise_analysis(electron_image: np.ndarray, det_cfg: dict,
                        output_path: str = None):
    """绘制噪声分析图（直方图 + 空间分布）。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 全图直方图
    ax = axes[0]
    flat = electron_image.flatten()
    ax.hist(flat, bins=200, color="steelblue", edgecolor="none", alpha=0.7)
    ax.set_xlabel("电子数 [e-]"); ax.set_ylabel("像元数")
    ax.set_title("像元电子数分布")
    ax.axvline(flat.mean(), color="r", linestyle="--", label=f"均值={flat.mean():.1f}")
    ax.legend(fontsize=8)

    # 空间图
    ax = axes[1]
    im = ax.imshow(electron_image, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="e-")
    ax.set_title("电子数空间分布")

    # 行剖面
    ax = axes[2]
    mid_row = electron_image.shape[0] // 2
    ax.plot(electron_image[mid_row, :], linewidth=1, color="darkorange")
    ax.set_xlabel("列像素"); ax.set_ylabel("电子数 [e-]")
    ax.set_title(f"中间行剖面（第{mid_row}行）")
    ax.grid(True, alpha=0.3)

    plt.suptitle("噪声分析", fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_aero_effect_comparison(image_before: np.ndarray, image_after: np.ndarray,
                                 aero_info: dict = None, output_path: str = None):
    """气动效应前后对比图。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im1 = axes[0].imshow(image_before, cmap="inferno", aspect="auto")
    plt.colorbar(im1, ax=axes[0], label="DN")
    axes[0].set_title("气动效应前")

    im2 = axes[1].imshow(image_after, cmap="inferno", aspect="auto")
    plt.colorbar(im2, ax=axes[1], label="DN")
    axes[1].set_title("气动效应后")

    diff = image_after.astype(np.float64) - image_before.astype(np.float64)
    im3 = axes[2].imshow(diff, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im3, ax=axes[2], label="ΔDN")
    axes[2].set_title("差值图（后 - 前）")

    title = "气动光学效应对比"
    if aero_info:
        r0 = aero_info.get("r0_cm", "N/A")
        r0_s = f"{r0:.2f}" if isinstance(r0, float) else r0
        title += f"  |  Fried r0={r0_s}cm"
        dr = aero_info.get("jitter_dr_px", 0)
        dc = aero_info.get("jitter_dc_px", 0)
        title += f"  抖动=({dr:.2f},{dc:.2f})px"

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_snr_vs_distance(cfg_base: dict, distances: np.ndarray,
                          output_path: str = None):
    """绘制信噪比随目标距离变化曲线。"""
    from modules.radiation import planck_integrated_radiance, compute_aperture_irradiance, get_atmospheric_transmittance
    from modules.detector import irradiance_to_electrons, NoiseModel

    snr_list = []
    ne_list  = []
    tau_list = []
    cfg = cfg_base
    rng = np.random.default_rng(0)
    noise_model = NoiseModel(cfg["detector"], rng)

    lmin = cfg["band"]["lambda_min"]
    lmax = cfg["band"]["lambda_max"]
    lam_c = (lmin + lmax) / 2.0
    T   = cfg["target"]["temperature"]
    eps = cfg["target"]["emissivity"]
    A   = cfg["target"]["area"]
    D   = cfg["optics"]["aperture_diameter"]
    t_op = cfg["optics"].get("transmission", 0.85)
    px  = cfg["detector"]["pixel_pitch"]
    t_int = cfg["detector"]["integration_time"]
    qe  = cfg["detector"]["quantum_efficiency"]
    band_name = cfg["band"].get("name", "MWIR")

    L_band = planck_integrated_radiance(T, lmin, lmax, eps)

    for R in distances:
        tau = get_atmospheric_transmittance(cfg["atmosphere"], R, band_name)
        E   = compute_aperture_irradiance(L_band, tau, A, R, D, t_op)
        Ne  = irradiance_to_electrons(E, px, t_int, qe, lam_c)
        sigma = noise_model.compute_noise_sigma(Ne)
        snr = Ne / (sigma + 1e-10)
        snr_list.append(snr)
        ne_list.append(Ne)
        tau_list.append(tau)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].semilogy(distances / 1000, snr_list, "b-", linewidth=2)
    axes[0].axhline(1, color="r", linestyle="--", label="SNR=1（检测阈值）")
    axes[0].set_xlabel("距离 [km]"); axes[0].set_ylabel("SNR")
    axes[0].set_title("信噪比 vs 距离"); axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].semilogy(distances / 1000, ne_list, "g-", linewidth=2)
    axes[1].set_xlabel("距离 [km]"); axes[1].set_ylabel("信号电子数 [e-]")
    axes[1].set_title("信号电子数 vs 距离"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(distances / 1000, tau_list, "m-", linewidth=2)
    axes[2].set_xlabel("距离 [km]"); axes[2].set_ylabel("大气透过率")
    axes[2].set_title("大气透过率 vs 距离"); axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"目标性能分析 | T={T}K, {band_name}", fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"SNR分析图已保存: {output_path}")
    plt.close()

    return {"distances_km": distances / 1000, "snr": snr_list,
            "N_e": ne_list, "tau": tau_list}