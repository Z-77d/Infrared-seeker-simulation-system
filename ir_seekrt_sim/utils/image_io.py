"""
工具模块：图像读写
==================
支持 16-bit TIFF / PNG / NPY 格式的红外图像存储。
"""

import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def save_image(
    image: np.ndarray,
    filepath: str,
    fmt: str = "tiff",
    normalize_for_preview: bool = False,
) -> str:
    """
    保存红外图像到文件。

    Parameters
    ----------
    image                  : 图像数据（uint16 或 float）
    filepath               : 输出路径（不含扩展名）
    fmt                    : "tiff" | "png" | "npy"
    normalize_for_preview  : 是否同时保存8-bit预览图

    Returns
    -------
    saved_path : 实际保存路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    ext_map = {"tiff": ".tiff", "tif": ".tiff", "png": ".png", "npy": ".npy"}
    ext = ext_map.get(fmt.lower(), ".tiff")
    full_path = filepath if filepath.endswith(ext) else filepath + ext

    try:
        if fmt.lower() in ("tiff", "tif"):
            import tifffile
            tifffile.imwrite(full_path, image.astype(np.uint16))
        elif fmt.lower() == "png":
            import cv2
            cv2.imwrite(full_path, image.astype(np.uint16))
        elif fmt.lower() == "npy":
            np.save(full_path, image)
        else:
            import tifffile
            tifffile.imwrite(full_path, image.astype(np.uint16))

        logger.info(f"图像已保存: {full_path}  shape={image.shape}  dtype={image.dtype}")

        # 保存8-bit预览图
        if normalize_for_preview:
            _save_preview(image, full_path)

    except ImportError as e:
        # 回退：用numpy保存
        npy_path = filepath + ".npy"
        np.save(npy_path, image)
        logger.warning(f"库不可用({e})，已回退保存为NPY: {npy_path}")
        full_path = npy_path

    return full_path


def _save_preview(image: np.ndarray, original_path: str):
    """生成并保存8-bit预览PNG。"""
    preview_path = original_path.rsplit(".", 1)[0] + "_preview.png"
    img_f = image.astype(np.float64)
    vmin, vmax = np.percentile(img_f, (1, 99))
    img_8 = np.clip((img_f - vmin) / (vmax - vmin + 1e-10) * 255, 0, 255).astype(np.uint8)

    try:
        import cv2
        cv2.imwrite(preview_path, img_8)
    except ImportError:
        # 用matplotlib保存
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(preview_path, img_8, cmap="inferno")

    logger.debug(f"预览图已保存: {preview_path}")


def load_image(filepath: str) -> np.ndarray:
    """
    加载红外图像。

    Parameters
    ----------
    filepath : 图像路径

    Returns
    -------
    image : numpy 数组
    """
    ext = Path(filepath).suffix.lower()
    if ext in (".tiff", ".tif"):
        import tifffile
        return tifffile.imread(filepath)
    elif ext == ".png":
        import cv2
        return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    elif ext == ".npy":
        return np.load(filepath)
    else:
        raise ValueError(f"不支持的图像格式: {ext}")


def normalize_to_8bit(image: np.ndarray, percentile_low: float = 1, percentile_high: float = 99) -> np.ndarray:
    """将任意图像归一化到0-255 uint8，用于可视化。"""
    img_f = image.astype(np.float64)
    vmin  = np.percentile(img_f, percentile_low)
    vmax  = np.percentile(img_f, percentile_high)
    img_n = np.clip((img_f - vmin) / (vmax - vmin + 1e-10), 0, 1)
    return (img_n * 255).astype(np.uint8)


def save_results_summary(results: dict, output_dir: str):
    """将仿真结果汇总保存为JSON。"""
    import json
    summary = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            summary[k] = f"ndarray shape={v.shape} dtype={v.dtype}"
        elif isinstance(v, (int, float, str, bool, list)):
            summary[k] = v
        else:
            summary[k] = str(v)

    path = os.path.join(output_dir, "simulation_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"仿真结果摘要已保存: {path}")
    return path