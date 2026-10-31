import pytest

from agent.groq_planner import GroqModelError, GroqModelPlanner


class _Delegate:
    def __init__(self):
        self.calls = []
        self.result = object()

    def plan(self, question, *, dataset_context, tool_catalog):
        self.calls.append(
            {
                "question": question,
                "dataset_context": dataset_context,
                "tool_catalog": tool_catalog,
            }
        )
        return self.result


def test_defaults_are_explicit_and_identify_groq():
    planner = GroqModelPlanner(client=object())

    assert planner.provider == "groq"
    assert planner.base_url == "https://api.groq.com/openai/v1"
    assert planner.model == "qwen/qwen3.8-27b"
    assert planner.parallel_tool_calls is True


def test_metadata_marks_groq_as_fallback_not_primary():
    metadata = GroqModelPlanner(client=object()).metadata()

    assert metadata["fallback_only"] is True
    assert metadata["primary_provider"] == "meta_model_api"
    assert metadata["provider"] == "groq"


def test_missing_key_is_rejected_when_no_client(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(GroqModelError):
        GroqModelPlanner()


def test_environment_key_allows_construction(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    planner = GroqModelPlanner()

    assert planner.model == "qwen/qwen3.8-27b"


def test_blank_model_is_rejected():
    with pytest.raises(GroqModelError):
        GroqModelPlanner(client=object(), model="   ")


def test_blank_base_url_is_rejected():
    with pytest.raises(GroqModelError):
        GroqModelPlanner(client=object(), base_url="   ")


def test_plan_delegates_without_changing_contract():
    planner = GroqModelPlanner(client=object())
    delegate = _Delegate()
    planner._planner = delegate

    result = planner.plan(
        "Compare three regions",
        dataset_context={"row_count": 120},
        tool_catalog=[{"name": "one_way_anova"}],
    )

    assert result is delegate.result
    assert delegate.calls == [
        {
            "question": "Compare three regions",
            "dataset_context": {"row_count": 120},
            "tool_catalog": [{"name": "one_way_anova"}],
        }
    ]
