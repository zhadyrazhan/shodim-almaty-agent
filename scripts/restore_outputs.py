"""Re-attach saved outputs after regenerating a notebook.

build_notebook.py emits a fresh skeleton with empty outputs, which would throw
away an executed run (the brief requires the notebook be submitted *with*
outputs). This matches cells by exact source text and copies the outputs and
execution_count back, so only genuinely new cells come back empty.

    python scripts/restore_outputs.py sxodim_agent.ipynb backup.ipynb
"""

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1

    target, source = Path(sys.argv[1]), Path(sys.argv[2])
    new = json.loads(target.read_text(encoding="utf-8"))
    old = json.loads(source.read_text(encoding="utf-8"))

    # Map source text -> (outputs, execution_count) from the executed notebook.
    saved = {
        "".join(c["source"]): (c.get("outputs", []), c.get("execution_count"))
        for c in old["cells"]
        if c["cell_type"] == "code" and c.get("outputs")
    }

    restored = 0
    for cell in new["cells"]:
        if cell["cell_type"] != "code":
            continue
        key = "".join(cell["source"])
        if key in saved:
            cell["outputs"], cell["execution_count"] = saved[key]
            restored += 1

    target.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")

    total = sum(1 for c in new["cells"] if c["cell_type"] == "code")
    print(f"restored outputs for {restored}/{total} code cells")
    if restored < len(saved):
        print(f"  note: {len(saved) - restored} saved cells no longer match (source changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
