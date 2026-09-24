from __future__ import annotations

from tools import cap_internal_force_analysis_tool as force_tool


class FakeOps:
    def __init__(self) -> None:
        self.nodes = {}
        self.elements = []
        self.reactions_assembled = False
        self.reaction_values = {}

    def wipe(self):
        return None

    def model(self, *_args):
        return None

    def geomTransf(self, *_args):
        return None

    def node(self, tag, x, y):
        self.nodes[tag] = (float(x), float(y))

    def element(self, *args):
        self.elements.append(args)

    def uniaxialMaterial(self, *_args):
        return None

    def fix(self, *_args):
        return None

    def nodeReaction(self, tag):
        return self.reaction_values.get(tag, [0.0, 0.0, 0.0])

    def reactions(self):
        self.reactions_assembled = True


def test_zero_length_support_nodes_have_identical_coordinates(monkeypatch) -> None:
    fake_ops = FakeOps()
    monkeypatch.setattr(force_tool, "ops", fake_ops)
    model_data = {
        "cap_length": 10.0,
        "cap_width": 2.0,
        "cap_height_mid": 1.5,
        "cantilever_length": 2.0,
        "column_spacing": 4.0,
        "column_count": 2,
        "superstructure_dead_load": {"support_points": []},
        "vehicle_live_load": {"input": {"supports": []}},
    }

    force_tool.build_model(model_data)

    zero_length_elements = [
        element for element in fake_ops.elements if element[0] == "zeroLength"
    ]
    assert len(zero_length_elements) == 2
    for _, _tag, pier_node, ground_node, *_args in zero_length_elements:
        assert fake_ops.nodes[pier_node] == fake_ops.nodes[ground_node]


def test_build_model_returns_column_supports(monkeypatch) -> None:
    fake_ops = FakeOps()
    monkeypatch.setattr(force_tool, "ops", fake_ops)
    model_data = {
        "cap_length": 10.0,
        "cap_width": 2.0,
        "cap_height_mid": 1.5,
        "cantilever_length": 2.0,
        "column_spacing": 4.0,
        "column_count": 2,
        "superstructure_dead_load": {"support_points": []},
        "vehicle_live_load": {"input": {"supports": []}},
    }

    _node_x_map, _ele_info, column_supports = force_tool.build_model(model_data)

    assert len(column_supports) == 2
    assert all("pier_node" in s and "ground_node" in s for s in column_supports)


def test_extract_column_reactions_returns_vertical_component(monkeypatch) -> None:
    fake_ops = FakeOps()
    fake_ops.reaction_values = {10: [1.0, 55.0, 3.0], 11: [2.0, 45.0, 4.0]}
    monkeypatch.setattr(force_tool, "ops", fake_ops)

    reactions = force_tool.extract_column_reactions([
        {"ground_node": 10},
        {"ground_node": 11},
    ])

    assert reactions == [55.0, 45.0]
    assert fake_ops.reactions_assembled is True
