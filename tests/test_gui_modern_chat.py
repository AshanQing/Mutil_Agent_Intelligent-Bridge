from bridge_agents.gui_modern_chat import ChatSession, format_answer


def test_chat_context_is_bounded_to_twelve_turns() -> None:
    session = ChatSession()
    for index in range(30):
        session.append("user", str(index))

    assert len(session.context()) == 24
    assert session.context()[0]["content"] == "6"
    assert session.context()[-1]["content"] == "29"


def test_chat_rejects_unknown_roles() -> None:
    session = ChatSession()

    try:
        session.append("system", "hidden")
    except ValueError as exc:
        assert "role" in str(exc)
    else:
        raise AssertionError("unknown role must fail")


def test_format_answer_appends_unique_artifact_and_code_citations() -> None:
    text = format_answer(
        {
            "answer": "盖梁控制截面满足要求。",
            "cited_artifact_ids": ["A-01", "A-01"],
            "cited_code_evidence_ids": ["JTG-3362-5.3.1"],
        }
    )

    assert text.startswith("盖梁控制截面满足要求。")
    assert text.count("A-01") == 1
    assert "JTG-3362-5.3.1" in text
