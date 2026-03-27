"""
单元测试模块
============
覆盖四个核心模块的主要功能验证。
运行方式：
    cd ir_seeker_sim
    python -m pytest tests/ -v
    或
    python tests/test_all.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import unittest


class TestRadiation(unittest.TestCase):
    """模块一：辐射传输测试"""

    def test_planck_blackbody(self):
        """验证 Planck 公式：Wien 位移定律峰值波长"""
        from modules.radiation import planck_spectral_radiance
        T = 1000.0  # K
        # Wien 位移定律：λ_max = 2898μm·K / T
        lam_peak_theory = 2898e-9 / T  # ~2.898μm
        lam = np.linspace(0.5e-6, 15e-6, 50000)
        L   = planck_spectral_radiance(lam, T)
        lam_peak_calc = lam[np.argmax(L)]
        rel_err = abs(lam_peak_calc - lam_peak_theory) / lam_peak_theory
        self.assertLess(rel_err, 0.01, f"Wien峰值误差: {rel_err:.4f}")

    def test_planck_monotone_with_temperature(self):
        """更高温度应有更高辐射"""
        from modules.radiation import planck_integrated_radiance
        L1 = planck_integrated_radiance(300, 3e-6, 5e-6)
        L2 = planck_integrated_radiance(500, 3e-6, 5e-6)
        self.assertGreater(L2, L1, "高温应有更高辐射亮度")

    def test_stefan_boltzmann_scaling(self):
        """验证 Stefan-Boltzmann：L ∝ T⁴（全波段积分）"""
        from modules.radiation import planck_integrated_radiance
        T1, T2 = 300.0, 600.0
        L1 = planck_integrated_radiance(T1, 0.1e-6, 100e-6, n_samples=2000)
        L2 = planck_integrated_radiance(T2, 0.1e-6, 100e-6, n_samples=2000)
        ratio_calc  = L2 / L1
        ratio_theory = (T2 / T1)**4
        rel_err = abs(ratio_calc - ratio_theory) / ratio_theory
        self.assertLess(rel_err, 0.05, f"Stefan-Boltzmann 误差: {rel_err:.4f}")

    def test_atmospheric_transmittance_range(self):
        """大气透过率应在 [0, 1]"""
        from modules.radiation import atmospheric_transmittance_exponential
        for dist in [0, 100, 1000, 10000, 100000]:
            tau = atmospheric_transmittance_exponential(dist, 2e-5)
            self.assertGreaterEqual(tau, 0.0)
            self.assertLessEqual(tau, 1.0)

    def test_irradiance_decreases_with_distance(self):
        """辐照度应随距离增大而减小"""
        from modules.radiation import planck_integrated_radiance, compute_aperture_irradiance
        L = planck_integrated_radiance(500, 3e-6, 5e-6)
        tau = 0.8
        E1 = compute_aperture_irradiance(L, tau, 1.0, 1000,  0.05)
        E2 = compute_aperture_irradiance(L, tau, 1.0, 10000, 0.05)
        self.assertGreater(E1, E2, "近距离辐照度应更大")


class TestDetector(unittest.TestCase):
    """模块二：探测器响应测试"""

    def _make_det_cfg(self):
        return {
            "array_rows": 64, "array_cols": 64,
            "pixel_pitch": 30e-6,
            "quantum_efficiency": 0.75,
            "responsivity": 2.0,
            "integration_time": 10e-3,
            "read_noise_electrons": 50.0,
            "dark_current": 1e4,
            "full_well_capacity": 2e6,
            "adc_bits": 14,
            "fpn_sigma": 0.01,
        }

    def test_electrons_positive(self):
        """电子数应为正值"""
        from modules.detector import irradiance_to_electrons
        det = self._make_det_cfg()
        Ne = irradiance_to_electrons(1e-3, det["pixel_pitch"],
                                      det["integration_time"],
                                      det["quantum_efficiency"], 4e-6)
        self.assertGreater(Ne, 0)

    def test_dn_range(self):
        """DN 值应在 [0, 2^bits-1]"""
        from modules.detector import electrons_to_dn
        det = self._make_det_cfg()
        electrons = np.array([0.0, 1e5, 1e6, 2e6, 3e6])
        dn = electrons_to_dn(electrons, det["full_well_capacity"], det["adc_bits"])
        self.assertTrue(np.all(dn >= 0))
        self.assertTrue(np.all(dn <= 2**det["adc_bits"] - 1))

    def test_noise_model_shape(self):
        """噪声模型输出形状应与输入一致"""
        from modules.detector import NoiseModel
        det = self._make_det_cfg()
        rng = np.random.default_rng(0)
        nm  = NoiseModel(det, rng)
        signal = np.ones((64, 64)) * 1e5
        result = nm.apply_noise(signal)
        self.assertEqual(result.shape, signal.shape)

    def test_noise_increases_variance(self):
        """添加噪声后方差应增大"""
        from modules.detector import NoiseModel
        det = self._make_det_cfg()
        rng = np.random.default_rng(42)
        nm  = NoiseModel(det, rng)
        signal = np.ones((64, 64)) * 1e5
        noisy  = nm.apply_noise(signal)
        self.assertGreater(np.std(noisy), np.std(signal))

    def test_fpn_map_shape(self):
        """FPN 图应与阵列尺寸匹配"""
        from modules.detector import NoiseModel
        det = self._make_det_cfg()
        nm  = NoiseModel(det)
        self.assertEqual(nm._fpn_map.shape, (64, 64))


class TestOptics(unittest.TestCase):
    """模块三：光学投影测试"""

    def _make_cfgs(self):
        optics = {"focal_length": 0.1, "aperture_diameter": 0.05,
                  "f_number": 2.0, "transmission": 0.85, "psf_model": "gaussian",
                  "psf_sigma_pixels": 1.2}
        det    = {"array_rows": 320, "array_cols": 240, "pixel_pitch": 30e-6,
                  "quantum_efficiency": 0.75, "integration_time": 10e-3,
                  "full_well_capacity": 2e6, "adc_bits": 14,
                  "read_noise_electrons": 50, "dark_current": 1e4, "fpn_sigma": 0.01}
        band   = {"lambda_min": 3e-6, "lambda_max": 5e-6}
        return optics, det, band

    def test_intrinsic_matrix_shape(self):
        from modules.optics import build_intrinsic_matrix
        K = build_intrinsic_matrix(0.1, 30e-6, 320, 240)
        self.assertEqual(K.shape, (3, 3))
        self.assertEqual(K[2, 2], 1.0)

    def test_intrinsic_matrix_principal_point(self):
        from modules.optics import build_intrinsic_matrix
        K = build_intrinsic_matrix(0.1, 30e-6, 320, 240)
        self.assertAlmostEqual(K[0, 2], 119.5)   # cx = (240-1)/2
        self.assertAlmostEqual(K[1, 2], 159.5)   # cy = (320-1)/2

    def test_on_axis_target_projects_to_center(self):
        """光轴目标应投影到图像中心附近"""
        from modules.optics import build_intrinsic_matrix, project_single_target
        K = build_intrinsic_matrix(0.1, 30e-6, 320, 240)
        row, col, in_fov = project_single_target([0, 0, 5000], K,
                                                   array_rows=320, array_cols=240)
        self.assertTrue(in_fov)
        self.assertAlmostEqual(col, 120, delta=2)
        self.assertAlmostEqual(row, 160, delta=2)

    def test_gaussian_psf_normalized(self):
        from modules.optics import gaussian_psf
        psf = gaussian_psf(1.5)
        self.assertAlmostEqual(psf.sum(), 1.0, places=5)

    def test_airy_psf_normalized(self):
        from modules.optics import airy_psf
        psf = airy_psf(0.05, 0.1, 30e-6, 4e-6)
        self.assertAlmostEqual(psf.sum(), 1.0, places=5)

    def test_render_target_increases_max(self):
        """渲染目标后图像最大值应增大"""
        from modules.optics import render_point_target, gaussian_psf
        bg = np.zeros((64, 64), dtype=np.float64)
        psf = gaussian_psf(1.2)
        result = render_point_target(bg, 32, 32, 1e5, psf)
        self.assertGreater(result.max(), bg.max())


class TestAeroOptics(unittest.TestCase):
    """模块四：气动光学效应测试"""

    def test_fried_parameter(self):
        """Fried参数应为正值"""
        from modules.aerooptics import compute_fried_parameter
        r0 = compute_fried_parameter(1e-14, 100.0, 4e-6)
        self.assertGreater(r0, 0)

    def test_phase_screen_shape(self):
        from modules.aerooptics import compute_fried_parameter, generate_phase_screen
        r0 = compute_fried_parameter(1e-14, 100.0, 4e-6)
        ps = generate_phase_screen(64, 0.05, r0)
        self.assertEqual(ps.shape, (64, 64))

    def test_phase_screen_to_psf(self):
        """相位屏转PSF应归一化且非负"""
        from modules.aerooptics import compute_fried_parameter, generate_phase_screen, phase_screen_to_psf
        r0 = compute_fried_parameter(1e-14, 100.0, 4e-6)
        ps = generate_phase_screen(64, 0.05, r0)
        psf = phase_screen_to_psf(ps)
        self.assertGreaterEqual(psf.min(), 0.0)
        self.assertAlmostEqual(psf.sum(), 1.0, places=4)

    def test_jitter_changes_image(self):
        """抖动应改变图像"""
        from modules.aerooptics import apply_image_jitter
        img = np.zeros((64, 64), dtype=np.uint16)
        img[32, 32] = 10000
        rng = np.random.default_rng(1)
        jittered, (dr, dc) = apply_image_jitter(img, 2.0, rng)
        self.assertFalse(np.array_equal(img, jittered))

    def test_heating_haze_preserves_dtype(self):
        """热层模糊应保持数据类型"""
        from modules.aerooptics import apply_aero_heating_haze
        img = (np.random.rand(64, 64) * 1000).astype(np.uint16)
        result = apply_aero_heating_haze(img, flow_velocity=500, path_length=100)
        self.assertEqual(result.dtype, img.dtype)

    def test_full_aero_pipeline(self):
        """完整气动效应流程不抛出异常"""
        from modules.aerooptics import apply_aero_optical_effects
        aero_cfg  = {"enable": True, "Cn2": 1e-14, "path_length": 100,
                     "L0": 10, "l0": 0.01, "jitter_sigma_pixels": 0.5,
                     "n_phase_screens": 2, "flow_velocity": 300}
        optics_cfg = {"aperture_diameter": 0.05, "focal_length": 0.1}
        band_cfg   = {"lambda_min": 3e-6, "lambda_max": 5e-6}
        det_cfg    = {"array_rows": 64, "array_cols": 64, "pixel_pitch": 30e-6}
        img = (np.random.rand(64, 64) * 1000).astype(np.uint16)
        rng = np.random.default_rng(42)
        result, info = apply_aero_optical_effects(img, aero_cfg, optics_cfg, band_cfg, det_cfg, rng)
        self.assertEqual(result.shape, img.shape)


class TestIntegration(unittest.TestCase):
    """集成测试：完整仿真流程"""

    def test_full_simulation(self):
        """完整仿真流程（使用默认配置）"""
        import yaml
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "config", "params.yaml")
        if not os.path.exists(cfg_path):
            self.skipTest("配置文件不存在")

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg["simulation"]["n_frames"] = 1
        cfg["aero_optics"]["n_phase_screens"] = 2

        from pipeline import simulate_frame
        rng = np.random.default_rng(0)
        results = simulate_frame(cfg, rng, frame_idx=0)

        self.assertIn("final_image", results)
        self.assertIn("E_aperture_Wm2", results)
        self.assertIn("NETD_K", results)
        self.assertGreater(results["E_aperture_Wm2"], 0)
        self.assertGreater(results["NETD_K"], 0)


if __name__ == "__main__":
    print("=" * 60)
    print("  红外导引头仿真系统 - 单元测试")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestRadiation, TestDetector, TestOptics, TestAeroOptics, TestIntegration]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)