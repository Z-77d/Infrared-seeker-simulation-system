# 红外导引头仿真系统

基于物理模型的红外导引头成像仿真，覆盖从目标辐射到最终图像的完整链路。

## 系统架构

```
① 辐射传输模型   →   ② 探测器响应模型   →   ③ 光学投影模型   →   ④ 气动光学效应
  Planck公式             噪声链                  针孔投影+PSF         湍流相位屏
  大气衰减               ADC量化                  图像渲染             视轴抖动
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行仿真（默认配置）
python pipeline.py

# 指定配置文件
python pipeline.py --config config/params.yaml --log-level DEBUG

# 运行单元测试
python tests/test_all.py
```

## 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| ① 辐射传输 | `modules/radiation.py` | Planck公式、大气透过率、入瞳辐照度 |
| ② 探测器响应 | `modules/detector.py` | 光子→电荷→DN、噪声链、NETD估算 |
| ③ 光学投影 | `modules/optics.py` | 针孔模型、PSF（高斯/Airy）、图像渲染 |
| ④ 气动效应 | `modules/aerooptics.py` | 湍流相位屏、热层模糊、图像抖动 |

## 主要物理模型

### ① 辐射传输
- **Planck谱辐射亮度**：$L_{bb}(\lambda,T)=\frac{2hc^2}{\lambda^5}\cdot\frac{1}{e^{hc/\lambda k_BT}-1}$
- **大气传输**：指数模型（Beer-Lambert）或 LOWTRAN 近似
- **入瞳辐照度**：$E=L_{band}\cdot A_{target}\cdot\tau_{atm}\cdot A_{aperture}/(\pi R^2)$

### ② 探测器响应
- **光子→电子**：$N_e = \eta\cdot(E\cdot A_{px}\cdot t_{int})/(hc/\lambda)$
- **噪声**：散粒噪声 + 读出噪声 + 暗电流 + 固定模式噪声（FPN）
- **NETD**：$NETD = \sigma_{noise} / (dN_e/dT)$

### ③ 光学投影
- **针孔模型**：$[u,v,1]^T=(1/Z)\cdot K\cdot X_{cam}$
- **PSF**：高斯（$\sigma$=1.2px）或 Airy盘（衍射极限）

### ④ 气动光学
- **Fried相干长度**：$r_0=0.185(\lambda^2/C_n^2 L)^{3/5}$
- **von Kármán谱**：$PSD(f)=0.023\cdot r_0^{-5/3}\cdot(f^2+f_0^2)^{-11/6}$
- 气动加热热层各向异性模糊 + 整体随机视轴抖动

## 配置参数

编辑 `config/params.yaml` 修改：
- 目标：温度、距离、面积、发射率
- 波段：MWIR（3-5μm）或 LWIR（8-12μm）
- 大气：透过率模型、能见度、湿度
- 光学：焦距、口径、PSF类型
- 探测器：阵列规格、量子效率、噪声参数
- 气动：$C_n^2$、湍流尺度、飞行速度

## 输出文件

```
output/
├── ir_image_frame000.tiff          # 16-bit 仿真图像（主输出）
├── ir_image_frame000_preview.png   # 8-bit 预览图
├── simulation_result_frame000.png  # 4模块结果对比图
└── simulation_summary.json         # 仿真参数摘要
```
