from slang_half_life import robustness as rb


def test_eligible_filters_placeholders_and_symbols():
    assert rb.eligible("rizz") and rb.eligible("OK boomer") and rb.eligible("lol")
    assert not rb.eligible("Unsupported titles/Space")
    assert not rb.eligible("+1") and not rb.eligible("C") and not rb.eligible("🧢")


def test_sample_is_reproducible_and_excludes_curated_terms():
    members = [f"word{i}" for i in range(1000)] + ["rizz", "Unsupported titles/x"]
    a = rb.draw_sample(members, exclude={"rizz"}, n=50, seed=42)
    b = rb.draw_sample(list(reversed(members)), exclude={"rizz"}, n=50, seed=42)
    assert a == b  # member order doesn't matter
    assert len(a) == 50 and "rizz" not in a
    assert rb.draw_sample(members, exclude=set(), n=50, seed=7) != a


def test_category_members_follows_continuation():
    pages = [
        {"query": {"categorymembers": [{"title": "a"}, {"title": "b"}]}, "continue": {"cmcontinue": "x"}},
        {"query": {"categorymembers": [{"title": "c"}]}},
    ]
    assert rb.category_members(fetch=lambda url: pages.pop(0)) == ["a", "b", "c"]
