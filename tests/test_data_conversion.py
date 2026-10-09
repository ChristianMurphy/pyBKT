"""convert_data numbers resources and lays out attempts the same way on every model type."""
import numpy as np
import pandas as pd
import pytest

from pyBKT.util import data_helper

# Student a answers x, y, x; student b answers y, y.
ATTEMPTS = pd.DataFrame(
    {
        "user_id": ["a", "a", "a", "b", "b"],
        "skill_name": "skill",
        "order_id": range(5),
        "template_id": ["x", "y", "x", "y", "y"],
        "correct": [1, 0, 1, 1, 0],
    }
)


def convert(model_type, **kwargs):
    return data_helper.convert_data(ATTEMPTS.copy(), "skill", model_type=model_type, **kwargs)["skill"]


def test_multipair_numbers_each_pair_in_order_of_first_appearance():
    data = convert([False, False, True, False])
    # A student's first attempt has no pair; then y after x, x after y, and y after y.
    assert data["resources"].tolist() == [1, 2, 3, 1, 4]
    assert sorted(data["resource_names"].values()) == [1, 2, 3, 4]
    assert data["resource_names"]["Default"] == 1


def test_multipair_reuses_fitted_pairs_and_rejects_new_ones():
    fitted = {"skill": convert([False, False, True, False])}
    again = convert([False, False, True, False], resource_refs=fitted)
    assert again["resources"].tolist() == [1, 2, 3, 1, 4]

    first_pairs_only = {"skill": {"resource_names": dict(list(fitted["skill"]["resource_names"].items())[:3])}}
    with pytest.raises(ValueError, match="not fitted"):
        convert([False, False, True, False], resource_refs=first_pairs_only)


def test_multiprior_puts_a_first_answer_slot_before_each_student():
    data = convert([False, True, False, False])
    assert data["data"].tolist() == [[0, 2, 1, 2, 0, 2, 1]]
    assert data["starts"].tolist() == [1, 5]
    assert data["lengths"].tolist() == [4, 3]
    # The slot's resource comes from the student's first answer, stored as 1 or 2.
    assert data["resources"].tolist() == [3, 1, 1, 1, 3, 1, 1]
    assert data["multiprior_index"].tolist() == [0, 4]


def test_multigs_gives_each_template_its_own_row():
    data = convert([False, False, False, True])
    assert data["gs_names"] == {"x": 0, "y": 1}
    assert data["data"].tolist() == [[2, 0, 2, 0, 0], [0, 1, 0, 2, 1]]


def test_skills_keep_their_rows_in_data_order():
    attempts = ATTEMPTS.assign(skill_name=["s1", "s2", "s1", "s2", "s1"])
    data = data_helper.convert_data(attempts, ".*")
    assert data["s1"]["index"].tolist() == [0, 2, 4]
    assert data["s2"]["index"].tolist() == [1, 3]
    assert data["s1"]["starts"].tolist() == [1, 3]
    assert data["s1"]["lengths"].tolist() == [2, 1]
