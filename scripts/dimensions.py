#!/usr/bin/env python3

from pathlib import Path


CIRCUITS = Path(__file__).resolve().parents[1] / "data" / "circuits"
DIMENSIONS = CIRCUITS / "dimensions.txt"


def aig_sizes(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        header = file.readline().decode("ascii").split()

    if len(header) < 6 or header[0] != "aig":
        raise ValueError(f"unsupported AIG header in {path}")

    inputs = int(header[2]) + int(header[3])
    outputs = int(header[4]) + (int(header[6]) if len(header) > 6 else 0)
    return inputs, outputs


def main() -> None:
    rows = []
    for path in sorted(CIRCUITS.glob("*/*.aig")):
        rows.append((f"{path.parent.name}/{path.stem}", *aig_sizes(path)))

    name_width = max(len(scheme_name) for scheme_name, _, _ in rows)
    input_width = max(len(str(inputs)) for _, inputs, _ in rows)
    output_width = max(len(str(outputs)) for _, _, outputs in rows)

    lines = []
    for scheme_name, inputs, outputs in rows:
        lines.append(
            f"{scheme_name:<{name_width}}: "
            f"i={inputs:>{input_width}} o={outputs:>{output_width}}"
        )
    DIMENSIONS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {DIMENSIONS.relative_to(CIRCUITS.parents[1])}")


if __name__ == "__main__":
    main()
