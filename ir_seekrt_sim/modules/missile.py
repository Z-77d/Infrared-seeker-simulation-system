"""
模块：导弹 3D 几何与热分布建模及渲染
功能：
1. 构建带属性的简单 3D 导弹多边形网格 (Fuselage, Nozzle, Plume, Fins)。
2. 3D 到 2D 坐标转换 (通过相机内参、外参投影)。
3. 使用画家算法 (Painter's Algorithm) 与 sub-pixel sub-rendering 在电子数图像上叠加。
"""

import numpy as np
import cv2
from scipy.signal import fftconvolve
import logging
from modules.radiation import planck_integrated_radiance
from modules.detector import power_to_electrons

logger = logging.getLogger(__name__)

class MissileTarget:
    """
    简化的导弹 3D 几何与热分布模型。
    坐标系定义（局部坐标系）：
        Z轴：指向飞行前方（机头正方向）。原点 Z=0 设在喷口与机身交界处。
        Y轴：指向下方。
        X轴：指向右侧（构成右手系）。
    """
    def __init__(self):
        self.faces = []
        self._build_geometry()

    def _add_cylinder(self, z_start, z_end, radius, temp, eps, name, sections=12):
        """围绕Z轴添加一个圆柱管的侧面多边形（四边形）"""
        angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
        for i in range(sections):
            a1, a2 = angles[i], angles[(i+1)%sections]
            pts = np.array([
                [radius * np.cos(a1), radius * np.sin(a1), z_start],
                [radius * np.cos(a2), radius * np.sin(a2), z_start],
                [radius * np.cos(a2), radius * np.sin(a2), z_end],
                [radius * np.cos(a1), radius * np.sin(a1), z_end],
            ])
            self.faces.append({"name": name, "pts_3d": pts, "temp": temp, "eps": eps})
            
    def _add_cone(self, z_start, z_end, r_start, r_end, temp, eps, name, sections=12):
        """围绕Z轴添加一个圆台侧面"""
        angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
        for i in range(sections):
            a1, a2 = angles[i], angles[(i+1)%sections]
            pts = np.array([
                [r_start * np.cos(a1), r_start * np.sin(a1), z_start],
                [r_start * np.cos(a2), r_start * np.sin(a2), z_start],
                [r_end * np.cos(a2),   r_end * np.sin(a2),   z_end],
                [r_end * np.cos(a1),   r_end * np.sin(a1),   z_end],
            ])
            self.faces.append({"name": name, "pts_3d": pts, "temp": temp, "eps": eps})

    def _add_fin(self, z_root_start, z_root_end, r_root, span, sweepback, angle, temp, eps, name):
        """添加一片十字翼面多边形"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        z1, z2 = z_root_start, z_root_end
        z3, z4 = z2 - sweepback, z1 - sweepback
        r_tip = r_root + span
        
        # 梯形翼节点（在Z-R平面上）
        pts_2d = [
            [r_root, z1],
            [r_root, z2],
            [r_tip, z3],
            [r_tip, z4]
        ]
        # 旋转到对应的角度
        pts_3d = np.array([[r * cos_a, r * sin_a, z] for r, z in pts_2d])
        self.faces.append({"name": name, "pts_3d": pts_3d, "temp": temp, "eps": eps})

    def _build_geometry(self):
        # 参数纯属演示示意（米、开尔文）
        # 机身：中温 T=340K，圆柱形。长度 3m，半径 0.15m。方向：Z从 0 到 3.0
        self._add_cylinder(0.0, 3.0, 0.15, temp=340.0, eps=0.9, name="Fuselage")
        
        # 尾喷口：高温 T=800K。圆台形。Z从 -0.2 到 0.0
        self._add_cone(-0.2, 0.0, 0.13, 0.15, temp=800.0, eps=0.95, name="Nozzle")
        
        # # 尾焰：极高温 T=1200K。假作锥形发光体，降低发射率以模拟半透明/弥散效果
        # self._add_cone(-2.5, -0.2, 0.02, 0.13, temp=1200.0, eps=0.3, name="Plume")
        # 尾焰：将温度从 1200K 降到 500K，观察能量溢出减少后的几何细节
        self._add_cone(-2.5, -0.2, 0.02, 0.13, temp=500.0, eps=0.3, name="Plume")
        
        # 尾翼：4面十字翼，略高于机身温度 T=320K。
        for i in range(4):
            self._add_fin(
                z_root_start=0.2, z_root_end=0.8, r_root=0.15, 
                span=0.4, sweepback=0.2, angle=i * np.pi / 2, 
                temp=320.0, eps=0.85, name=f"Fin_{i}"
            )

def render_3d_missile_target(
    bg_electron_image: np.ndarray,
    K: np.ndarray,
    R_cam: np.ndarray,
    t_cam: np.ndarray,
    band_cfg: dict,
    atm_tau: float,
    optics_cfg: dict,
    det_cfg: dict,
    psf: np.ndarray,
    missile: MissileTarget = None
) -> np.ndarray:
    """
    计算并在焦平面的背景图上叠加渲染 3D 导弹目标，并与 PSF 卷积。
    """
    if missile is None:
        missile = MissileTarget()
        
    rows, cols = bg_electron_image.shape
    # 使用全0初始画布，用于独立绘制目标层，避免把背景噪声也跟着目标再卷积一遍
    target_img = np.zeros((rows, cols), dtype=np.float64)
    
    lam_min = band_cfg["lambda_min"]
    lam_max = band_cfg["lambda_max"]
    lam_c   = (lam_min + lam_max) / 2.0
    
    # 提取光学与探测器参数
    f_num = optics_cfg.get("f_number", 2.0)
    t_opt = optics_cfg.get("transmission", 0.85)
    
    px = det_cfg["pixel_pitch"]
    t_int = det_cfg["integration_time"]
    qe = det_cfg["quantum_efficiency"]
    A_px = px ** 2

    projected_faces = []
    
    # 1. 坐标系转换与背面剔除
    for face in missile.faces:
        pts = face["pts_3d"]
        
        # 外参变换: X_cam = R_cam * X_world + t_cam
        X_cam = (R_cam @ pts.T).T + t_cam
        
        # 法线与视线计算 (背面剔除)
        v1 = X_cam[1] - X_cam[0]
        v2 = X_cam[2] - X_cam[0]
        n_cross = np.cross(v1, v2)
        norm_n = np.linalg.norm(n_cross)
        if norm_n == 0:
            continue
        n_dir = n_cross / norm_n
        view_dir = -X_cam[0]  # 近似视线向量
        
        # 对于不透明实体表面，如果背对相机则剔除；尾焰(Plume)视作半透明发射体保留
        if np.dot(n_dir, view_dir) < 0 and face["name"] != "Plume":
            continue
            
        avg_z = np.mean(X_cam[:, 2])
        if avg_z <= 0:
            continue  # 在相机后方
            
        # 投影到像素坐标
        uv_h = (K @ X_cam.T).T
        uv   = uv_h[:, :2] / uv_h[:, 2:]
        
        # 2. 物理辐射量计算
        # 计算该面辐射亮度
        L_band = planck_integrated_radiance(face["temp"], lam_min, lam_max, face["eps"])
        # 在焦平面上产生的辐照度 (由于立体角投影关系，面源在焦平面的辐照度仅由其实际辐射亮度和 F数决定!)
        #公式: E_fp = L_band * tau_atm * t_opt * (pi / (4 * F#^2))
        E_fp = L_band * atm_tau * t_opt * np.pi / (4.0 * f_num**2)
        
        # 计算单像元接收到的功率及相应电子数
        P_px = E_fp * A_px
        N_e_px = power_to_electrons(P_px, t_int, qe, lam_c)
        
        projected_faces.append({
            "uv": uv,
            "z_depth": avg_z,
            "ne": N_e_px,
            "name": face["name"]
        })
        
    # 3. 按深度排序（从远到近 - 画家算法）
    projected_faces.sort(key=lambda x: x["z_depth"], reverse=True)
    
    # 4. 亚像素多边形光栅化 (Sub-pixel rendering via OpenCV)
    # 利用 cv2.fillPoly 的 shift 机制 (4 bit = 16x 精度亚像素)
    shift = 4
    scale = 1 << shift
    
    for face in projected_faces:
        pts_2d = face["uv"]
        color_val = float(face["ne"])
        
        # 生成定点化坐标
        pts_scaled = (pts_2d * scale).astype(np.int32)
        # 用抗锯齿画法完美填充亚像素边缘
        cv2.fillPoly(target_img, [pts_scaled], color=color_val, lineType=cv2.LINE_AA, shift=shift)

    # 5. 经过光学系统 PSF 带来的衍射与像差弥散卷积
    target_spread = fftconvolve(target_img, psf, mode="same")
    
    # 提取有目标的遮罩，进行简单的合并（背景被不透明弹体遮挡处的原始噪声覆盖）
    # 在这个简单物理模型里，直接把衍射后的目标能量叠加上去即可呈现完美的视觉效果
    final_image = bg_electron_image + target_spread
    return final_image
