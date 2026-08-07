"""Tests voor gui.nav — navigatiemodel, pure logica."""

from gui import nav


def test_wizard_flow_steps_are_sequential():
    steps = [item.step for item in nav.WIZARD_FLOW]
    assert steps == list(range(1, len(steps) + 1))


def test_all_items_includes_home_flow_and_tools():
    routes = {item.route for item in nav.all_items()}
    assert nav.HOME.route in routes
    assert "/config" in routes
    assert "/benchmark" in routes


def test_tools_have_no_step():
    assert all(item.step is None for item in nav.TOOLS)


def test_register_and_query_availability():
    assert nav.is_available("/") is True  # home altijd beschikbaar
    fake = "/__test_only__"
    assert nav.is_available(fake) is False
    nav.register_route(fake)
    assert nav.is_available(fake) is True
