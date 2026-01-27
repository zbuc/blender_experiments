# Repository Guidelines

This repo is a Blender-based tooling sandbox for silhouette-driven 3D blockouts,
image-based validation, and performance experiments. Most workflows run inside
Blender's bundled Python, with pure-Python benchmarks for hotspots.

## Repository layout
- `blender_blocking/`: main tool code, tests, and integration scripts
- `blender_blocking/benchmarks/benchmark_perf.py`: micro-benchmarks
- `blender_blocking/test_images/`: sample input silhouettes
- `blender_blocking/test_output/`: test outputs (renders, logs)
- `configs/`: JSON configs for each reconstruction mode and quality tier
- `scripts/`: utility scripts (config generation, debug helpers)

## Setup
Install dependencies into Blender's Python (preferred), and make them visible to tests:

**Windows (PowerShell):**
```powershell
$py = "C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe"
$target = "$env:USERPROFILE\blender_python_packages"
& $py -m pip install --target $target -r "blender_blocking\requirements.txt"
```

**macOS (bash/zsh):**
```bash
# Adjust path to your Blender installation
PY="/Applications/Blender.app/Contents/Resources/5.0/python/bin/python3.11"
TARGET="$HOME/blender_python_packages"
$PY -m pip install --target $TARGET -r blender_blocking/requirements.txt
```

Most test runners already add `~/blender_python_packages` to `sys.path`.

## Run the full test suite
**Windows (PowerShell):**
```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" `
  --background --python (Join-Path $PWD 'blender_blocking\test_runner.py')
```

**macOS (bash/zsh):**
```bash
/Applications/Blender.app/Contents/MacOS/Blender \
  --background --python blender_blocking/test_runner.py
```

## E2E validation (image -> mesh -> render -> IoU)
Run with default settings (sample images):

**Windows (PowerShell):**
```powershell
& $blender --background --python blender_blocking\test_e2e_validation.py
```

**macOS (bash/zsh):**
```bash
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
$BLENDER --background --python blender_blocking/test_e2e_validation.py
```

Run with custom images:

**Windows (PowerShell):**
```powershell
& $blender --background --python blender_blocking\test_e2e_validation.py -- `
  --reconstruction-mode silhouette_intersection `
  --config-path configs\silhouette_intersection-default.json `
  --front <front.png> --side <side.png> --top <top.png>
```

**macOS (bash/zsh):**
```bash
$BLENDER --background --python blender_blocking/test_e2e_validation.py -- \
  --reconstruction-mode silhouette_intersection \
  --config-path configs/silhouette_intersection-default.json \
  --front <front.png> --side <side.png> --top <top.png>
```

Outputs land in `blender_blocking/test_output/e2e_renders` with filenames like:
`<input>_<mode>_<config>_<view>_<n>.png`

Progress bars are on by default. Use `--no-progress` to disable.

### Party mode (fun render lighting/materials)
Party mode works only when `force_material` is false:

**Windows (PowerShell):**
```powershell
& $blender --background --python blender_blocking\test_e2e_validation.py -- `
  --config-json '{"render_silhouette":{"party_mode":true,"force_material":false}}'
```

**macOS (bash/zsh):**
```bash
$BLENDER --background --python blender_blocking/test_e2e_validation.py -- \
  --config-json '{"render_silhouette":{"party_mode":true,"force_material":false}}'
```

## Configs and quality tiers
Configs live in `configs/` with names like:
- `<mode>-default.json`
- `<mode>-high-quality.json`
- `<mode>-ultra.json`
- `<mode>-extreme-ultra.json`

Regenerate the defaults:

**Windows (PowerShell):**
```powershell
python scripts\generate_default_configs.py
```

**macOS (bash/zsh):**
```bash
python3 scripts/generate_default_configs.py
```

Key reconstruction options in configs:
- `reconstruction.reconstruction_mode`: `legacy`, `loft_profile`, `silhouette_intersection`
- `silhouette_intersection`: `extrude_distance`, `contour_mode`, `boolean_solver`,
  `silhouette_extract_override`, `largest_component_only`
- `render_silhouette`: `resolution`, `samples`, `engine`, `margin_frac`,
  `color_mode`, `force_material`, `background_color`, `silhouette_color`,
  `camera_distance_factor`, `party_mode`

## Silhouette-intersection debug helper
Run the dedicated debug script to inspect meshes and booleans:

**Windows (PowerShell):**
```powershell
& $blender --background --python scripts\debug_silhouette_intersection.py -- `
  --base blender_blocking\test_images\car --config-path configs\silhouette_intersection-default.json
```

**macOS (bash/zsh):**
```bash
$BLENDER --background --python scripts/debug_silhouette_intersection.py -- \
  --base blender_blocking/test_images/car --config-path configs/silhouette_intersection-default.json
```

## Benchmarks
Run all benches:

**Windows (PowerShell):**
```powershell
python blender_blocking\benchmarks\benchmark_perf.py --all
```

**macOS (bash/zsh):**
```bash
python3 blender_blocking/benchmarks/benchmark_perf.py --all
```

Targeted run with higher settings:

**Windows (PowerShell):**
```powershell
python blender_blocking\benchmarks\benchmark_perf.py --bench visual_hull `
  --resolution 128 --num-views 360 --include-top --repeat 3
```

**macOS (bash/zsh):**
```bash
python3 blender_blocking/benchmarks/benchmark_perf.py --bench visual_hull \
  --resolution 128 --num-views 360 --include-top --repeat 3
```

Common flags:
- `--iterations`, `--resolution`, `--num-views`, `--include-top`, `--repeat`
- `--profile-size`, `--profile-samples`, `--combine-profiles`, `--slice-profiles`
- `--resfit-points`, `--resfit-primitives`, `--resfit-full-steps`, `--resfit-opt-steps`
- `--progress` / `--no-progress`
