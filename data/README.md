# data/

This folder holds the baseline dataset used to model "normal" household
energy consumption, before any attacks are simulated on top of it.

**Dataset:** UCI Individual Household Electric Power Consumption
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

2,075,259 minute-level measurements from one household in France over
almost 4 years (2006-2010) — real consumption patterns, not synthetic.

## To fetch it (Day 2)

```
pip install ucimlrepo
```

```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=235)
df = dataset.data.features
```

Raw data files are intentionally **not** committed to this repo (see
`.gitignore`) — the file is 130MB+ uncompressed and trivially reproducible
from the source above, so committing it would just bloat the repo for no
reason.
