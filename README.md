# Infrared Seeker Imaging Simulation System

A physics-based infrared seeker imaging simulation framework covering the full imaging chain from target radiation to the final image.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Simulation

Run with the default configuration:

```bash
python pipeline.py
```

Run with a specified configuration file:

```bash
python pipeline.py --config config/params.yaml --log-level DEBUG
```

Run unit tests:

```bash
python tests/test_all.py
```

## Modules

| Module | File | Function |
|------|------|------|
| Radiative Transfer | `modules/radiation.py` | Planck-law radiation, atmospheric transmittance, entrance pupil irradiance |
| Detector Response | `modules/detector.py` | Photon-to-charge-to-DN conversion, noise chain, NETD estimation |
| Optical Projection | `modules/optics.py` | Pinhole model, PSF (Gaussian/Airy), image rendering |
| Aero-Optical Effects | `modules/aerooptics.py` | Turbulence phase screen, thermal-layer blur, image jitter |

## Output Files

```text
output/
├── ir_image_frame000.tiff          # 16-bit simulated image (main output)
├── ir_image_frame000_preview.png   # 8-bit preview image
├── simulation_result_frame000.png  # comparison figure of the four modules
└── simulation_summary.json         # summary of simulation parameters
```
