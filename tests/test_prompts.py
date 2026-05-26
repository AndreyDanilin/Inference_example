from pathlib import Path

from inference_lab.prompts import load_prompt_suite


def test_load_prompt_suite_reads_named_chat_prompts(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text(
        """
prompts:
  - name: short
    scenario: short
    messages:
      - role: user
        content: Say hello in five words.
  - name: long
    scenario: long
    messages:
      - role: system
        content: You are concise.
      - role: user
        content: Explain continuous batching.
""".strip(),
        encoding="utf-8",
    )

    prompts = load_prompt_suite(prompt_file)

    assert [prompt.name for prompt in prompts] == ["short", "long"]
    assert prompts[0].messages == [{"role": "user", "content": "Say hello in five words."}]
    assert prompts[1].scenario == "long"
