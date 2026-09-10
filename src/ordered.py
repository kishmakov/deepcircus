"""Sampled restriction lattices with at most two postponed query coordinates."""

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.config import OrderedConfig


def _remove_equivalent_query_columns(coordinates: np.ndarray) -> np.ndarray:
    patterns = np.packbits(coordinates ^ coordinates[:1], axis=0, bitorder="little").T
    _, columns = np.unique(patterns, axis=0, return_index=True)
    columns.sort()
    nonconstant = np.any(coordinates[:, columns] != coordinates[:1, columns], axis=0)
    return coordinates[:, columns[nonconstant]]


def _set_bit_indices(mask: int) -> list[int]:
    indices = []
    while mask:
        indices.append((mask & -mask).bit_length() - 1)
        mask &= mask - 1
    return indices


def _make_ordering_prefixes(dimensions: int, orders: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    prefixes = []
    for _ in range(orders):
        prefix = [0]
        for axis in rng.permutation(dimensions).tolist():
            prefix.append(prefix[-1] | (1 << axis))
        prefixes.append(prefix)
    return prefixes


def _shortest_covering_prefix(mask: int, prefix: list[int]) -> int:
    low, high = mask.bit_count(), len(prefix) - 1
    while low < high:
        middle = (low + high) // 2
        if mask & ~prefix[middle]:
            low = middle + 1
        else:
            high = middle
    return low


def _queries_entering_ordering(mask: int, prefix: list[int]) -> int:
    length = _shortest_covering_prefix(mask, prefix)
    holes = length - mask.bit_count()
    if holes > 3:
        return 0
    # Filling one of three holes can bring the child into this ordering.
    available = prefix[length] & ~mask
    if holes < 3:
        extension = min(len(prefix) - 1, length + 3 - holes)
        available |= prefix[extension] & ~prefix[length]
    return available


class _QueryChoices:
    def __init__(self, dimensions: int, orders: int, seed: int):
        self.prefixes = _make_ordering_prefixes(dimensions, orders, seed)
        self.by_mask: dict[int, list[int]] = {}

    def candidates(self, mask: int) -> list[int]:
        if mask not in self.by_mask:
            available = 0
            for prefix in self.prefixes:
                available |= _queries_entering_ordering(mask, prefix)
            self.by_mask[mask] = _set_bit_indices(available)
        return self.by_mask[mask]


def _make_empty_layer() -> dict:
    return dict(nodes=[None], zero=[], one=[], parent=[], split=[])


def _index_one_cells(coordinates: np.ndarray) -> list[int]:
    return [sum(1 << cell for cell in np.flatnonzero(coordinates[:, axis]).tolist())
            for axis in range(coordinates.shape[1])]


class _TopologyBuilder:
    def __init__(self, coordinates: np.ndarray, orders: int, seed: int):
        self.layers = [_make_empty_layer() for _ in range(coordinates.shape[1] + 1)]
        self.queries = _QueryChoices(coordinates.shape[1], orders, seed)
        self.ones = _index_one_cells(coordinates)
        self.visited: dict[tuple[int, int], int] = {}

    def visit_restriction(self, mask: int, members: int) -> int:
        if not members:
            return 0
        key = (mask, members)
        if key in self.visited:
            return self.visited[key]
        layer = self.layers[mask.bit_count()]
        index = len(layer["nodes"])
        self.visited[key] = index
        if members.bit_count() == 1:
            layer["nodes"].append(members.bit_length() - 1)
            return index
        layer["nodes"].append(None)
        for axis in self.queries.candidates(mask):
            self._add_query_children(layer, index, mask, members, axis)
        assert layer["parent"][-1] == index, (mask, members)
        return index

    def _add_query_children(self, layer: dict, parent: int, mask: int, members: int, axis: int) -> None:
        high = members & self.ones[axis]
        low = members ^ high
        child_mask = mask | (1 << axis)
        zero = self.visit_restriction(child_mask, low)
        one = self.visit_restriction(child_mask, high)
        layer["zero"].append(zero)
        layer["one"].append(one)
        layer["parent"].append(parent)
        layer["split"].append(bool(low and high))


def _representative_child_pairs(layer: dict) -> tuple[np.ndarray, np.ndarray]:
    count = len(layer["nodes"])
    zero, one = np.zeros(count, np.int64), np.zeros(count, np.int64)
    for parent, a, b in zip(layer["parent"], layer["zero"], layer["one"]):
        zero[parent], one[parent] = a, b
    return zero, one


def _finalize_layer_tensors(layer: dict, device: torch.device) -> dict:
    layer["first_zero"], layer["first_one"] = _representative_child_pairs(layer)
    layer["leaf"] = [cell is not None for cell in layer["nodes"]]
    layer["cell"] = [0 if cell is None else cell for cell in layer["nodes"]]
    layer["count"] = len(layer["nodes"])
    for key in ("zero", "one", "parent", "first_zero", "first_one", "cell"):
        layer[key] = torch.tensor(layer[key], dtype=torch.long, device=device)
    for key in ("split", "leaf"):
        layer[key] = torch.tensor(layer[key], dtype=torch.bool, device=device)
    del layer["nodes"]
    return layer


class OrderedTopology:
    """Device indices for the restriction graph; construction state is discarded."""

    # Cached indices and masks must remain usable by autograd after validation.
    @torch.inference_mode(False)
    def __init__(self, coordinates: np.ndarray, orders: int, seed: int, device: torch.device):
        coordinates = _remove_equivalent_query_columns(coordinates)
        cells, self.dimensions = coordinates.shape
        builder = _TopologyBuilder(coordinates, orders, seed)
        self.root = builder.visit_restriction(0, (1 << cells) - 1)
        self.node_count = len(builder.visited)
        self.mask_count = len(builder.queries.by_mask)
        self.layers = [_finalize_layer_tensors(layer, device) for layer in builder.layers]


def _unpack_sampled_points(packed: torch.Tensor, points: int, bitness: int) -> np.ndarray:
    point_dim = 3 * bitness + 2
    rows = np.unpackbits(packed.detach().cpu().numpy(), axis=1, bitorder="little")
    return rows[:, :points * point_dim].reshape(-1, points, point_dim)


def _group_cases_by_input_set(rows: np.ndarray, bitness: int) -> dict:
    groups = {}
    for case, row in enumerate(rows):
        inputs, indices = np.unique(row[:, :bitness], axis=0, return_index=True)
        key = inputs.tobytes()
        if key not in groups:
            groups[key] = (inputs, [])
        values = row[indices][:, [bitness, 2 * bitness + 1]]
        groups[key][1].append((case, values))
    return groups


def _expand_helper_coordinates(inputs: np.ndarray) -> np.ndarray:
    zero = np.concatenate((inputs, np.zeros((len(inputs), 1), np.uint8)), 1)
    one = np.concatenate((inputs, np.ones((len(inputs), 1), np.uint8)), 1)
    return np.concatenate((zero, one), 0)


def _propagate_target_presence(present: torch.Tensor, observed: torch.Tensor, layer: dict) -> torch.Tensor:
    parents = torch.maximum(present[:, layer["first_zero"]], present[:, layer["first_one"]])
    return torch.where(layer["leaf"][None, :, None], observed[:, layer["cell"]], parents)


def _minimum_query_states(values: torch.Tensor, layer: dict) -> torch.Tensor:
    states = values.new_full((len(values), layer["count"], values.shape[-1]), float("inf"))
    parents = layer["parent"][None, :, None].expand_as(values)
    return states.scatter_reduce(1, parents, values, reduce="amin", include_self=True)


class OrderedRestrictionPredictor(nn.Module):
    """Learn branch combinations and root scores over a sampled restriction graph."""

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

    def _topology_for_input_set(self, key: bytes, inputs: np.ndarray, device: torch.device) -> OrderedTopology:
        if key != self._topology_key:
            coordinates = _expand_helper_coordinates(inputs) if self.model_name == "m1" else inputs
            self._topology = OrderedTopology(coordinates, self.config.orders, self.config.order_seed, device)
            self._topology_key = key
        return self._topology

    def _observe_target_values(self, tables: torch.Tensor) -> torch.Tensor:
        g, f = tables.unbind(-1)
        observed = torch.stack(((1 - g) * f, g * f), -1)
        if self.model_name == "m1":
            helper_zero = torch.stack(((1 - g) * (1 - f), g * (1 - f)), -1)
            observed = torch.cat((helper_zero, observed), 1)
        return observed

    def _combine_query_children(self, hidden: torch.Tensor, layer: dict) -> torch.Tensor:
        a, b = hidden[:, layer["zero"]], hidden[:, layer["one"]]
        values = self.combine(torch.cat((torch.maximum(a, b), a + b), -1))
        # A coordinate constant on this sampled group needs no query.
        return torch.where(layer["split"][None, :, None], values, torch.maximum(a, b))

    def _forward_layer(self, hidden: torch.Tensor, active: torch.Tensor, layer: dict) -> torch.Tensor:
        if len(layer["zero"]):
            values = self._combine_query_children(hidden, layer)
            next_hidden = _minimum_query_states(values, layer)
        else:
            next_hidden = hidden.new_full((len(hidden), layer["count"], self.hidden), float("inf"))
        return torch.where(active[..., None], next_hidden, 0.0)

    def _checkpoint_layer_update(self, hidden: torch.Tensor, active: torch.Tensor, layer: dict) -> torch.Tensor:
        if torch.is_grad_enabled():
            # Retain this layer's inputs and recompute branch activations during backward.
            return checkpoint(self._forward_layer, hidden, active, layer,
                              use_reentrant=False, preserve_rng_state=False)
        return self._forward_layer(hidden, active, layer)

    def forward_values(self, tables: torch.Tensor, topology: OrderedTopology) -> torch.Tensor:
        observed = self._observe_target_values(tables)
        hidden = tables.new_zeros((len(tables), 1, self.hidden))
        present = tables.new_zeros((len(tables), 1, 2))
        for layer in reversed(topology.layers):
            present = _propagate_target_presence(present, observed, layer)
            active = present.prod(-1) > 0
            hidden = self._checkpoint_layer_update(hidden, active, layer)
        return self.bitness - self.head(hidden[:, topology.root])

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        rows = _unpack_sampled_points(packed, self.points, self.bitness)
        groups = _group_cases_by_input_set(rows, self.bitness)
        output = torch.empty((len(rows), 2), dtype=torch.float32, device=packed.device)
        for key, (inputs, cases) in groups.items():
            topology = self._topology_for_input_set(key, inputs, packed.device)
            values = torch.tensor(np.stack([row for _, row in cases]), dtype=torch.float32, device=packed.device)
            output[[case for case, _ in cases]] = self.forward_values(values, topology)
        return output
