"""Tests for framedex.parsing — the pure parsing/normalization helpers."""

from framedex.parsing import (
    coerce_people_count,
    is_permission_denied,
    pick_diar_auth_kwarg,
)

# --- coerce_people_count ---------------------------------------------------


def test_coerce_people_count_plain_int() -> None:
    assert coerce_people_count(5, 0) == 5


def test_coerce_people_count_clamps_high() -> None:
    assert coerce_people_count(500, 0) == 99


def test_coerce_people_count_clamps_negative() -> None:
    assert coerce_people_count(-3, 0) == 0


def test_coerce_people_count_bool_is_zero() -> None:
    # bool is an int subclass — must not become 1
    assert coerce_people_count(True, 0) == 0


def test_coerce_people_count_float_truncates() -> None:
    assert coerce_people_count(4.9, 0) == 4


def test_coerce_people_count_numeric_string() -> None:
    assert coerce_people_count("12", 0) == 12


def test_coerce_people_count_tilde_prefix() -> None:
    assert coerce_people_count("~15", 0) == 15


def test_coerce_people_count_many_uses_face_count() -> None:
    # "many" → face_count as the lower bound, with a floor of 10
    assert coerce_people_count("many", 20) == 20
    assert coerce_people_count("many", 3) == 10


def test_coerce_people_count_few_and_some() -> None:
    assert coerce_people_count("few", 0) == 2
    assert coerce_people_count("some", 0) == 3


def test_coerce_people_count_none_words() -> None:
    assert coerce_people_count("none", 0) == 0
    assert coerce_people_count("", 0) == 0


def test_coerce_people_count_junk_is_zero() -> None:
    assert coerce_people_count("bananas", 0) == 0
    assert coerce_people_count(None, 0) == 0


# --- is_permission_denied --------------------------------------------------


def test_is_permission_denied_short_telltale() -> None:
    assert is_permission_denied("I need permission to continue.") is True


def test_is_permission_denied_case_insensitive() -> None:
    assert is_permission_denied("PLEASE GRANT access first") is True


def test_is_permission_denied_real_description() -> None:
    assert is_permission_denied("A wide shot of a giraffe at sunrise.") is False


def test_is_permission_denied_long_text_not_flagged() -> None:
    # A long, legitimate description that merely mentions permission
    text = "I need permission " + "x" * 600
    assert is_permission_denied(text) is False


# --- pick_diar_auth_kwarg --------------------------------------------------


def test_pick_diar_auth_kwarg_prefers_token() -> None:
    assert pick_diar_auth_kwarg(["self", "token", "device"]) == "token"


def test_pick_diar_auth_kwarg_legacy_name() -> None:
    assert pick_diar_auth_kwarg(["self", "use_auth_token"]) == "use_auth_token"


def test_pick_diar_auth_kwarg_both_prefers_token() -> None:
    assert pick_diar_auth_kwarg(["token", "use_auth_token"]) == "token"


def test_pick_diar_auth_kwarg_neither_defaults_token() -> None:
    assert pick_diar_auth_kwarg([]) == "token"
    assert pick_diar_auth_kwarg(["self", "device"]) == "token"
