from miniharness.events import safe_arguments_preview


def test_safe_arguments_preview_redacts_sensitive_values():
    preview = safe_arguments_preview(
        {
            "path": "calculator.py",
            "api_key": "top-secret",
            "options": {
                "authorization": "Bearer secret-token",
                "mode": "check",
            },
        }
    )

    assert preview["path"] == "calculator.py"
    assert preview["api_key"] == "[REDACTED]"
    assert "Bearer secret-token" not in preview["options"]
    assert "[REDACTED]" in preview["options"]


def test_safe_arguments_preview_is_bounded():
    preview = safe_arguments_preview(
        {
            "content": "x" * 200,
            **{
                f"value_{index}": index
                for index in range(10)
            },
        },
        max_items=3,
        max_value_length=12,
    )

    assert len(preview) == 4
    assert preview["..."] == "8 more argument(s)"
    assert len(preview["content"]) == 12
    assert preview["content"].endswith("…")


def test_safe_arguments_preview_is_deterministic():
    first = safe_arguments_preview(
        {
            "z": 1,
            "a": ["pytest", "-q"],
        }
    )
    second = safe_arguments_preview(
        {
            "a": ["pytest", "-q"],
            "z": 1,
        }
    )

    assert first == second
    assert list(first) == ["a", "z"]
