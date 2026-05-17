# Neural CPM — Standalone Teaching Package

Walkthrough of the **Neural Closest-Point Method** for PDEs on surfaces, plus the classical 2-D CPM demo.

## Contents

```
neural-cpm/
├── ncpm_poisson_and_heat.ipynb   teaching notebook (main entry point)
├── data/                         Apple_surface_feature.npy, pretrained weights
├── model/                        SurfNO neural operator
├── utils/                        all helpers (band, Laplacian, RBF, ...)
├── precomputed/                  for web vis
└── web/                          local HTML+JS demo (classical 2-D + 3-D Apple)
```


## Setup

```bash
conda env create -f environment.yml
conda activate mapsenv
cd neural_cpm
```

## Run the notebook

Best for openning with vscode
Also ok:
```bash
jupyter lab ncpm_poisson_and_heat.ipynb
```

## Run the web demo

```bash
cd neural-cpm
python -m http.server 8000           # serve from neural-cpm/  (NOT from web/)
# open http://localhost:8000/web/index.html
```

The HTML page contains:

1. The **2-D classical CPM** demo.
2. The 3-D **Neural CPM** results on the Apple example for Poisson and Heat.
