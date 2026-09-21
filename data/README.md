# Dataset placement

This project uses the **HELECAR-D** analysed driving records. The dataset is **not** versioned in this repository. Download it locally, then point the code at the analysed CSV files.

## Official sources

- Dataset paper: NaitMalek, Y., Najib, M., Bakhouya, M., Gaber, J., 2023. HELECAR-D: A dataset for urban electro mobility in Moroccan context. *Data in Brief* 48, 109080. https://doi.org/10.1016/j.dib.2023.109080
- Data release: https://doi.org/10.5281/zenodo.7217707 (https://zenodo.org/record/7217707)
- License: CC-BY-4.0

## Expected location

Place the download so analysed trip CSVs are discoverable from the repository root. Discovery is recursive and does not assume a published folder layout.

Typical layout:

```
PINN_ev_energy/
├── HELECAR-D/data/Analysed/T1/*.csv
├── HELECAR-D/data/Analysed/T2/*.csv
└── HELECAR-D/data/Analysed/T3/*.csv
```

The root `HELECAR-D/` directory is gitignored. Analysed CSVs are preferred over raw CAN dumps.

## Do not

- Rename or overwrite original CSVs in place.
- Invent missing trips or labels.
- Point the code at a substitute dataset.
- Commit raw or analysed HELECAR-D files.

If no HELECAR-D CSVs are found, experiment scripts refuse to invent paper metrics.
