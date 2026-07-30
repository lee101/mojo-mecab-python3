from __future__ import annotations

import MeCab
import numpy as np
import pytest

import mojo_mecab as mm


def node_records(node):
    records = []
    while node is not None:
        records.append(
            (
                node.surface,
                node.feature,
                node.length,
                node.stat,
                node.isbest,
                node.wcost,
                node.cost,
            )
        )
        node = node.next
    return records


def best_surfaces(text):
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)
    surfaces = []
    while node is not None:
        if node.stat in (MeCab.MECAB_NOR_NODE, MeCab.MECAB_UNK_NODE):
            surfaces.append(node.surface)
        node = node.next
    return tuple(surfaces)


def test_compatibility_classes_and_version_are_upstream_objects():
    assert mm.Tagger is MeCab.Tagger
    assert mm.Model is MeCab.Model
    assert mm.Lattice is MeCab.Lattice
    assert mm.Node is MeCab.Node
    assert mm.VERSION == MeCab.VERSION == "0.996"


def test_every_advertised_upstream_name_is_the_upstream_object():
    for name in MeCab.__all__:
        assert getattr(mm, name) is getattr(MeCab, name)


@pytest.mark.parametrize(
    "args",
    ["", "-Owakati", "-Odump", "-N 3"],
)
def test_tagger_parse_exact_parity(args):
    text = "pythonが大好きです。"
    assert mm.Tagger(args).parse(text) == MeCab.Tagger(args).parse(text)


def test_parse_to_node_exact_field_parity():
    text = "庭には二羽鶏がいる"
    assert node_records(mm.Tagger().parseToNode(text)) == node_records(
        MeCab.Tagger().parseToNode(text)
    )


def test_parse_nbest_exact_parity():
    text = "今日の天気"
    assert mm.Tagger().parseNBest(4, text) == MeCab.Tagger().parseNBest(4, text)


def test_published_wakati_example():
    assert mm.Tagger("-Owakati").parse("pythonが大好きです").split() == [
        "python",
        "が",
        "大好き",
        "です",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "すももももももものうち",
        "pythonが大好きです",
        "庭には二羽鶏がいる",
        "未知語XYZです",
        "東京スカイツリーへ行った。",
        "これはペンです。This is a pen.",
        "空白 を 含む 文",
    ],
)
def test_mojo_recomputes_every_upstream_node_cost(text):
    lattice = mm.mecab_lattice(text)
    result = mm.decode(lattice)
    np.testing.assert_array_equal(result.costs, lattice.upstream_costs)
    assert result.cost == int(lattice.upstream_costs[-1])
    assert mm.decoded_surfaces(lattice, result) == best_surfaces(text)


def test_model_and_lattice_api_remain_usable():
    model = mm.Model()
    tagger = model.createTagger()
    lattice = model.createLattice()
    lattice.set_sentence("形態素解析")
    assert tagger.parse(lattice)
    assert lattice.is_available()
    assert lattice.toString().endswith("EOS\n")
