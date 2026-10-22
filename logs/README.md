# logs/

Four structured JSON-line logs, written by `src/logger_config.py`. Log files
are not committed (see `.gitignore`) because they are generated output.
Run `python main.py` to regenerate.

| File | Question it answers |
|---|---|
| `detection.log` | What was flagged, by which detector, with what score |
| `operational.log` | Is the pipeline healthy, or silently failing |
| `audit.log` | What threshold or config changed, when, and why |
| `ground_truth.log` | What attacks were actually injected, for scoring |

Kept separate rather than combined so an investigation can ask one question
without filtering noise from the other three. Every line is a single JSON
object, so the logs load directly into pandas or a SIEM with no parsing code.
