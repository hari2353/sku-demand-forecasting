"""CI verification: check the portfolio forecast run produced valid plans."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    summary = json.loads(Path("artifacts/portfolio_summary.json").read_text(encoding="utf-8"))
    assert len(summary["per_sku"]) > 0, "no SKUs selected"
    assert all(s["selected_model"] for s in summary["per_sku"].values()), "missing model"
    assert summary["overall"]["smape"] > 0, "invalid overall sMAPE"
    models = summary["overall"]["models_selected"]
    assert sum(models.values()) == len(summary["per_sku"]), "selection count mismatch"
    print(
        "portfolio forecast check passed for",
        len(summary["per_sku"]),
        "SKUs:",
        models,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
