# Dataset placement

Download **HELECAR-D** (NaitMalek et al., *Data in Brief*, 2023; CC-BY-4.0) and place it so analysed trip CSVs are discoverable from the repository root. Discovery is recursive and does not assume a published folder layout.

Typical layout:

```
HELECAR-D/data/Analysed/T1/*.csv
HELECAR-D/data/Analysed/T2/*.csv
HELECAR-D/data/Analysed/T3/*.csv
```

Analysed CSVs are preferred over raw CAN dumps. The root `HELECAR-D/` directory is gitignored and must not be committed. Code and paper artifacts remain in this repository; the dataset retains its own license.

## Do not

- Rename or overwrite original CSVs in place.
- Invent missing trips or labels.
- Point the code at a substitute dataset.
- Commit raw or analysed HELECAR-D files.

If no HELECAR-D CSVs are found, experiment scripts refuse to invent paper metrics.
