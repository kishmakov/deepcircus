"""Sampled restriction lattices with at most two postponed query coordinates."""

import numpy as np
import torch
from torch import nn

from src.config import OrderedConfig


class OrderedTopology:
    def __init__(self, coordinates: np.ndarray, orders: int, seed: int, device: torch.device):
        # Constant and duplicate/complemented columns describe the same sampled query.
        patterns = np.packbits(coordinates ^ coordinates[:1], axis=0, bitorder="little").T
        _, columns = np.unique(patterns, axis=0, return_index=True)
        columns.sort()
        columns = columns[np.any(coordinates[:, columns] != coordinates[:1, columns], axis=0)]
        coordinates = coordinates[:, columns]
        cells, dimensions = coordinates.shape
        self.dimensions = dimensions
        self.layers = [dict(nodes=[None], zero=[], one=[], parent=[], split=[])
                       for _ in range(dimensions + 1)]
        rng = np.random.default_rng(seed)
        prefixes = []
        for _ in range(orders):
            prefix = [0]
            for axis in rng.permutation(dimensions).tolist():
                prefix.append(prefix[-1] | (1 << axis))
            prefixes.append(prefix)
        choices, visited = {}, {}
        ones = [sum(1 << i for i in np.flatnonzero(coordinates[:, axis]).tolist())
                for axis in range(dimensions)]

        def candidates(mask):
            if mask in choices:
                return choices[mask]
            used, available = mask.bit_count(), 0
            for prefix in prefixes:
                low, high = used, dimensions
                while low < high:
                    middle = (low + high) // 2
                    if mask & ~prefix[middle]:
                        low = middle + 1
                    else:
                        high = middle
                holes = low - used
                # A child may enter this ordering even if its parent has three holes.
                if holes <= 3:
                    available |= prefix[low] & ~mask
                    if holes < 3:
                        available |= prefix[min(dimensions, low + 3 - holes)] & ~prefix[low]
            result = []
            while available:
                result.append((available & -available).bit_length() - 1)
                available &= available - 1
            choices[mask] = result
            return result

        def visit(mask, members):
            if not members:
                return 0
            key = (mask, members)
            if key in visited:
                return visited[key]
            layer = self.layers[mask.bit_count()]
            index = len(layer["nodes"])
            visited[key] = index
            if members.bit_count() == 1:
                layer["nodes"].append(members.bit_length() - 1)
                return index
            layer["nodes"].append(None)
            for axis in candidates(mask):
                high = members & ones[axis]
                low = members ^ high
                zero = visit(mask | (1 << axis), low)
                one = visit(mask | (1 << axis), high)
                layer["zero"].append(zero)
                layer["one"].append(one)
                layer["parent"].append(index)
                layer["split"].append(bool(low and high))
            assert layer["parent"][-1] == index, (mask, members)
            return index

        self.root = visit(0, (1 << cells) - 1)
        self.node_count = len(visited)
        self.mask_count = len(choices)
        for layer in self.layers:
            count = len(layer["nodes"])
            zero, one = np.zeros(count, np.int64), np.zeros(count, np.int64)
            for parent, a, b in zip(layer["parent"], layer["zero"], layer["one"]):
                zero[parent], one[parent] = a, b
            layer["first_zero"], layer["first_one"] = zero, one
            layer["leaf"] = [cell is not None for cell in layer["nodes"]]
            layer["cell"] = [0 if cell is None else cell for cell in layer["nodes"]]
            layer["count"] = count
            for key in ("zero", "one", "parent", "first_zero", "first_one", "cell"):
                layer[key] = torch.tensor(layer[key], dtype=torch.long, device=device)
            for key in ("split", "leaf"):
                layer[key] = torch.tensor(layer[key], dtype=torch.bool, device=device)
            del layer["nodes"]


class OrderedRestrictionPredictor(nn.Module):
    """A fixed number of orderings gives O(points * bitness^3) sampled states."""

    def __init__(self, bitness: int, points: int, model_name: str, config: OrderedConfig):
        super().__init__()
        assert bitness >= 1, bitness
        assert points > 1, points
        assert model_name in ("m1", "m2"), model_name
        self.bitness, self.points, self.model_name = bitness, points, model_name
        self.config, self.hidden = config, config.hidden
        self._topology_key, self._topology = None, None
        self.combine = nn.Sequential(
            nn.Linear(2 * config.hidden, config.branch_hidden), nn.ReLU(),
            nn.Linear(config.branch_hidden, config.hidden), nn.Softplus(),
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden, config.head_hidden), nn.ReLU(), nn.Linear(config.head_hidden, 2),
        )

    def forward_values(self, tables: torch.Tensor, topology: OrderedTopology) -> torch.Tensor:
        g, f = tables.unbind(-1)
        observed = torch.stack(((1 - g) * f, g * f), -1)
        if self.model_name == "m1":
            zero = torch.stack(((1 - g) * (1 - f), g * (1 - f)), -1)
            observed = torch.cat((zero, observed), 1)
        cases = len(tables)
        hidden = tables.new_zeros((cases, 1, self.hidden))
        present = tables.new_zeros((cases, 1, 2))
        for layer in reversed(topology.layers):
            next_present = torch.maximum(present[:, layer["first_zero"]], present[:, layer["first_one"]])
            next_present = torch.where(layer["leaf"][None, :, None], observed[:, layer["cell"]], next_present)
            next_hidden = tables.new_full((cases, layer["count"], self.hidden), float("inf"))
            if len(layer["zero"]):
                a, b = hidden[:, layer["zero"]], hidden[:, layer["one"]]
                values = self.combine(torch.cat((torch.maximum(a, b), a + b), -1))
                # A coordinate constant on this sampled group needs no query.
                values = torch.where(layer["split"][None, :, None], values, torch.maximum(a, b))
                parent = layer["parent"][None, :, None].expand_as(values)
                next_hidden = next_hidden.scatter_reduce(1, parent, values, reduce="amin", include_self=True)
            hidden = torch.where((next_present.prod(-1) > 0)[..., None], next_hidden, 0.0)
            present = next_present
        return self.bitness - self.head(hidden[:, topology.root])

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        n = self.bitness
        rows = np.unpackbits(packed.detach().cpu().numpy(), axis=1, bitorder="little")
        rows = rows[:, :self.points * (3 * n + 2)].reshape(-1, self.points, 3 * n + 2)
        groups = {}
        for case, row in enumerate(rows):
            inputs, indices = np.unique(row[:, :n], axis=0, return_index=True)
            key = inputs.tobytes()
            if key not in groups:
                groups[key] = (inputs, [])
            groups[key][1].append((case, row[indices][:, [n, 2 * n + 1]]))
        output = torch.empty((len(rows), 2), dtype=torch.float32, device=packed.device)
        for key, (inputs, cases) in groups.items():
            if key != self._topology_key:
                coordinates = inputs
                if self.model_name == "m1":
                    zero = np.concatenate((inputs, np.zeros((len(inputs), 1), np.uint8)), 1)
                    one = np.concatenate((inputs, np.ones((len(inputs), 1), np.uint8)), 1)
                    coordinates = np.concatenate((zero, one), 0)
                self._topology = OrderedTopology(coordinates, self.config.orders, self.config.order_seed, packed.device)
                self._topology_key = key
            values = torch.tensor(np.stack([row for _, row in cases]), dtype=torch.float32, device=packed.device)
            output[[case for case, _ in cases]] = self.forward_values(values, self._topology)
        return output
