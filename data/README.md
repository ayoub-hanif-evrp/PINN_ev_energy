# Place HELECAR-D here (optional)

A copy of **HELECAR-D** (NaitMalek et al., *Data in Brief*, 2023; CC-BY-4.0) is included at `HELECAR-D/`. Discovery is recursive and does not assume a published folder layout.

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

Analysed CSVs are preferred over raw CAN dumps.

## Do not

- Rename or overwrite original CSVs in place.
- Invent missing trips or labels.
- Point the code at a substitute dataset.

If no HELECAR-D CSVs are found, unit tests still run on synthetic fixtures under `tests/fixtures/`. Experiment scripts will not fabricate paper results.
