from pathlib import Path


CIRCUITS = Path(__file__).resolve().parent / "circuits"


def aig_sizes(path):
    with path.open("rb") as file:
        header = file.readline().decode("ascii").split()

    if len(header) < 6 or header[0] != "aig":
        raise ValueError(f"unsupported AIG header in {path}")

    inputs = int(header[2]) + int(header[3])
    outputs = int(header[4]) + (int(header[6]) if len(header) > 6 else 0)
    return inputs, outputs


def main():
    rows = []
    for path in sorted(CIRCUITS.glob("*/*.aig")):
        rows.append((f"{path.parent.name}/{path.stem}", *aig_sizes(path)))

    name_width = max(len(scheme_name) for scheme_name, _, _ in rows)
    input_width = max(len(str(inputs)) for _, inputs, _ in rows)
    output_width = max(len(str(outputs)) for _, _, outputs in rows)

    for scheme_name, inputs, outputs in rows:
        print(
            f"{scheme_name:<{name_width}}: "
            f"i={inputs:>{input_width}} o={outputs:>{output_width}}"
        )


if __name__ == "__main__":
    main()
