import csv
from pathlib import Path

class CsvLogger:
    def __init__(self, path: str, n_stages: int = 1):
        self.path   = Path(path)
        self.fields = self._make_csv_fields(n_stages)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._file   = open(self.path, mode="a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fields)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def _make_csv_fields(self, n_stages: int = 1) -> list:
        base = [
            "step",
            "split",
            "wall_time",
            "lm_loss",
            "perplexity",
            "lb_loss",
            "ratio_loss",    # Lratio agregado (eq. 10)
            "total_loss",
            "bpb",           # bits-per-byte
            "bpic",          # bytes-per-innermost-chunk
        ]
        compression = [f"compression_L{s+1}/L{s}" for s in range(n_stages)]
        tail = ["lr", "tokens_per_sec"]
        return base + compression + tail

    def log(self, **kwargs):
        row = {k: kwargs.get(k, "") for k in self.fields}
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()
