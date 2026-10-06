# NMDC LiDAR Driver View — Slightly More Realistic

This is a small rendering refinement of the previous pothole-detectable program.

The detection logic remains LiDAR-based. The main changes are:
- stronger driver-view perspective/depth,
- smoother continuous road surface,
- local elevation shading,
- softer horizon/fog gradient,
- less CAD-like obstacle blocks,
- rounded/shaded road-defect geometry,
- subtle foreground depth shading.

It still does not display raw LiDAR points or object names.

Run:
```powershell
pip install -r requirements.txt
python lidar_driver_view.py --input your_scan.bin --output driver_view_realistic.png
```
