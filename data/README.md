# Place HELECAR-D here (optional)

The experiment code **recursively searches the project** for candidate CSV files. You do **not** have to use this folder if the dataset already lives elsewhere in the project (for example `HELECAR-D/data/Analysed/`).

## Preferred layout

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
