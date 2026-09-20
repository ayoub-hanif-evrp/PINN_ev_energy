# Place HELECAR-D here (optional)

Download the official **HELECAR-D** release (NaitMalek et al., *Data in Brief*, 2023; CC-BY-4.0) onto this machine. Do **not** commit the dataset into Git. Discovery is recursive and does not assume a published folder layout.

Typical local layouts:

```
data/HELECAR-D/data/Analysed/T1/*.csv
HELECAR-D/data/Analysed/T1/*.csv
```

```
data/
  HELECAR-D/
    data/
      Analysed/
        T1/*.csv
        T2/*.csv
        T3/*.csv
      raw/
        ...
```

Analysed CSVs are preferred over raw CAN dumps. The root `HELECAR-D/` directory is gitignored.

## Do not

- Rename or overwrite original CSVs in place.
- Invent missing trips or labels.
- Point the code at a substitute dataset.
- Commit raw or analysed HELECAR-D files.

If no HELECAR-D CSVs are found, unit tests still run on synthetic fixtures under `tests/fixtures/`. Experiment scripts will not fabricate paper results.
