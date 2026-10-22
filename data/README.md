# data/

Holds the baseline consumption dataset. Raw data files are not committed
(see `.gitignore`) because the UCI file is over 130MB and reproducible from
source, so committing it would bloat the repo for no benefit.

## Default: calibrated synthetic data

`src/data_loader.py` generates synthetic hourly data by default, matched to
the UCI dataset's statistical profile (mean 0.98 kW against UCI's 1.09, with
the same daily and weekly structure). No download needed, so the repo runs
immediately on clone.

## Real data: UCI Individual Household Electric Power Consumption

2,075,259 minute-level measurements from one household in France, December
2006 to November 2010.

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

```bash
pip install ucimlrepo
```

Then in `main.py` swap `generate_synthetic_data(days=90)` for
`load_uci_dataset()`. Identical schema, so nothing downstream changes.

## Other datasets worth knowing about

- **Pecan Street Dataport**: real smart meter data from many homes with
  appliance-level breakdown. Requires academic registration.
- **UMass Smart* Dataset**: three homes, includes solar generation. Useful
  for extending this to solar inverter attacks.
