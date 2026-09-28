from kev.uc51.semantic_core import bind_typed_relation, execute


def test_legacy_binding_uses_typed_slots_not_a_color_vocabulary():
    binding = bind_typed_relation("COPY_VALUE", "Copy ultraviolet to ochre")
    assert binding["kind"] == "COPY_VALUE"
    assert binding["slots"]["source"]["value"] == "ultraviolet"
    assert binding["slots"]["destination"]["value"] == "ochre"
    assert binding["tool_execution"] == "NONE"


def test_legacy_execute_is_a_non_executing_typed_compatibility_alias():
    binding = execute("MAGNITUDE", "Magnitude of -7")
    assert binding["slots"]["input"] == {
        "type": "number",
        "value": -7,
        "raw": "-7",
    }
    assert binding["authority"] == "NONE"
