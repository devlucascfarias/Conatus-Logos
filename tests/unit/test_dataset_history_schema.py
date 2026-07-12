"""Testes da extensão de schema pra multi-turno (D-dataset-history-schema,
docs/plan_dataset_expansion_wave2_identity_multiturn.md seção 4) — campo opcional
`trajectory.history`, validação via `validate_trajectory`, e o suporte real em
`src.harness.Trajectory` pra construir/renderizar exemplos com histórico."""

from src.dataset.pipeline import structural_validate
from src.dataset.schema import validate_trajectory
from src.harness import HistoryTurn, Trajectory


def _base_metadata(id_="t1", task_type="single_tool_call"):
    return {
        "id": id_,
        "domain": "test",
        "language": "python",
        "difficulty": "easy",
        "tools_used": [],
        "num_steps": 0,
        "task_type": task_type,
        "source": "curated_manual",
        "license": "synthetic-no-license-needed",
        "validation_status": "pending",
        "execution_performed": False,
    }


# --- validate_trajectory ------------------------------------------------------------


def test_trajectory_without_history_is_valid():
    trajectory = {"system_prompt": "sp", "user_request": "oi", "raw_text": "<final>ok</final>"}
    assert validate_trajectory(trajectory) == []


def test_trajectory_with_empty_history_is_valid():
    trajectory = {"system_prompt": "sp", "user_request": "oi", "raw_text": "<final>ok</final>", "history": []}
    assert validate_trajectory(trajectory) == []


def test_trajectory_with_valid_history_is_valid():
    trajectory = {
        "system_prompt": "sp",
        "user_request": "segundo pedido",
        "raw_text": "<final>ok</final>",
        "history": [{"user_request": "primeiro pedido", "raw_text": "<final>primeira resposta</final>"}],
    }
    assert validate_trajectory(trajectory) == []


def test_trajectory_with_multiple_history_turns_is_valid():
    trajectory = {
        "system_prompt": "sp",
        "user_request": "terceiro pedido",
        "raw_text": "<final>ok</final>",
        "history": [
            {"user_request": "primeiro", "raw_text": "<final>r1</final>"},
            {"user_request": "segundo", "raw_text": "<final>r2</final>"},
        ],
    }
    assert validate_trajectory(trajectory) == []


def test_history_item_missing_raw_text_is_invalid():
    trajectory = {
        "system_prompt": "sp",
        "user_request": "oi",
        "raw_text": "<final>ok</final>",
        "history": [{"user_request": "sem resposta registrada"}],
    }
    problems = validate_trajectory(trajectory)
    assert problems != []


def test_history_item_missing_user_request_is_invalid():
    trajectory = {
        "system_prompt": "sp",
        "user_request": "oi",
        "raw_text": "<final>ok</final>",
        "history": [{"raw_text": "<final>resposta sem pedido registrado</final>"}],
    }
    problems = validate_trajectory(trajectory)
    assert problems != []


def test_history_not_a_list_is_invalid():
    trajectory = {
        "system_prompt": "sp",
        "user_request": "oi",
        "raw_text": "<final>ok</final>",
        "history": "não é uma lista",
    }
    problems = validate_trajectory(trajectory)
    assert problems != []


def test_missing_raw_text_is_invalid():
    # raw_text sempre foi lido direto (`example["trajectory"]["raw_text"]`) antes dessa
    # extensão — ausência disso agora vira erro de validação claro em vez de KeyError.
    trajectory = {"system_prompt": "sp", "user_request": "oi"}
    problems = validate_trajectory(trajectory)
    assert problems != []


# --- structural_validate integra validate_trajectory --------------------------------


def test_structural_validate_accepts_valid_history():
    example = {
        "metadata": _base_metadata(),
        "trajectory": {
            "system_prompt": "sp",
            "user_request": "segundo pedido",
            "raw_text": "<final>segunda resposta</final>",
            "history": [{"user_request": "primeiro pedido", "raw_text": "<final>primeira resposta</final>"}],
        },
    }
    assert structural_validate(example) == []


def test_structural_validate_rejects_malformed_history():
    example = {
        "metadata": _base_metadata(),
        "trajectory": {
            "system_prompt": "sp",
            "user_request": "oi",
            "raw_text": "<final>ok</final>",
            "history": [{"user_request": "sem raw_text"}],
        },
    }
    problems = structural_validate(example)
    assert any(p.startswith("trajectory: ") for p in problems)


def test_structural_validate_missing_raw_text_gives_clean_error_not_keyerror():
    example = {"metadata": _base_metadata(), "trajectory": {"system_prompt": "sp", "user_request": "oi"}}
    problems = structural_validate(example)
    assert any(p.startswith("trajectory: ") for p in problems)


# --- Trajectory.render_for_model com history -----------------------------------------


def test_render_for_model_without_history_matches_previous_behavior():
    traj = Trajectory(system_prompt="SP", user_request="oi", raw_text="<final>ok</final>")
    assert traj.render_for_model() == "SP\n\n[USER]\noi\n\n[ASSISTANT]\n<final>ok</final>"


def test_render_for_model_prepends_history_in_user_assistant_format():
    traj = Trajectory(
        system_prompt="SP",
        user_request="segundo pedido",
        raw_text="<final>segunda resposta</final>",
        history=[HistoryTurn(user_request="primeiro pedido", raw_text="<think>pensei</think><final>primeira resposta</final>")],
    )
    expected = (
        "SP"
        "\n\n[USER]\nprimeiro pedido\n\n[ASSISTANT]\n<think>pensei</think><final>primeira resposta</final>"
        "\n\n[USER]\nsegundo pedido\n\n[ASSISTANT]\n<final>segunda resposta</final>"
    )
    assert traj.render_for_model() == expected


def test_render_for_model_with_multiple_history_turns_preserves_order():
    traj = Trajectory(
        system_prompt="SP",
        user_request="terceiro",
        raw_text="<final>r3</final>",
        history=[
            HistoryTurn(user_request="primeiro", raw_text="<final>r1</final>"),
            HistoryTurn(user_request="segundo", raw_text="<final>r2</final>"),
        ],
    )
    rendered = traj.render_for_model()
    assert rendered.index("primeiro") < rendered.index("segundo") < rendered.index("terceiro")


# --- Trajectory.to_example_dict --------------------------------------------------------


def test_to_example_dict_omits_history_key_when_empty():
    traj = Trajectory(system_prompt="SP", user_request="oi", raw_text="<final>ok</final>")
    body = traj.to_example_dict()
    assert "history" not in body
    assert body == {"system_prompt": "SP", "user_request": "oi", "raw_text": "<final>ok</final>"}


def test_to_example_dict_includes_history_when_present():
    traj = Trajectory(
        system_prompt="SP",
        user_request="segundo",
        raw_text="<final>r2</final>",
        history=[HistoryTurn(user_request="primeiro", raw_text="<final>r1</final>")],
    )
    body = traj.to_example_dict()
    assert body["history"] == [{"user_request": "primeiro", "raw_text": "<final>r1</final>"}]


def test_to_example_dict_output_passes_validate_trajectory():
    traj = Trajectory(
        system_prompt="SP",
        user_request="segundo",
        raw_text="<final>r2</final>",
        history=[HistoryTurn(user_request="primeiro", raw_text="<final>r1</final>")],
    )
    assert validate_trajectory(traj.to_example_dict()) == []


# --- D-prompt-environment-block: bloco de ambiente (SO/shell) no prompt --------------


def test_render_for_model_without_environment_matches_previous_behavior():
    # Sem environment, o formato tem que ser byte-a-byte idêntico ao de antes (nenhum exemplo
    # legado pode mudar de tokenização).
    traj = Trajectory(system_prompt="SP", user_request="oi", raw_text="<final>ok</final>")
    assert traj.render_for_model() == "SP\n\n[USER]\noi\n\n[ASSISTANT]\n<final>ok</final>"


def test_render_for_model_inserts_environment_block_after_system_prompt():
    traj = Trajectory(
        system_prompt="SP", user_request="lista os arquivos", raw_text="<final>ok</final>",
        environment={"os": "Windows", "shell": "powershell"},
    )
    expected = (
        "SP"
        "\n\n<environment>\nos: Windows\nshell: powershell\n</environment>"
        "\n\n[USER]\nlista os arquivos\n\n[ASSISTANT]\n<final>ok</final>"
    )
    assert traj.render_for_model() == expected


def test_render_for_model_environment_comes_before_history():
    traj = Trajectory(
        system_prompt="SP", user_request="segundo", raw_text="<final>r2</final>",
        environment={"os": "Linux"},
        history=[HistoryTurn(user_request="primeiro", raw_text="<final>r1</final>")],
    )
    rendered = traj.render_for_model()
    assert rendered.index("<environment>") < rendered.index("primeiro") < rendered.index("segundo")


def test_to_example_dict_includes_environment_when_present():
    traj = Trajectory(
        system_prompt="SP", user_request="oi", raw_text="<final>ok</final>",
        environment={"os": "macOS", "shell": "bash"},
    )
    body = traj.to_example_dict()
    assert body["environment"] == {"os": "macOS", "shell": "bash"}
    assert validate_trajectory(body) == []


def test_to_example_dict_omits_environment_when_absent():
    traj = Trajectory(system_prompt="SP", user_request="oi", raw_text="<final>ok</final>")
    assert "environment" not in traj.to_example_dict()


def test_environment_block_is_in_masked_prefix_not_trained_tokens():
    # O bloco de ambiente é contexto: deve entrar no PREFIXO (mascarado), nunca nos tokens
    # treinados do turno atual. `_render_prefix` (data_collator) reusa render_for_model com
    # raw_text vazio, então o env fica antes do ponto onde raw_text começaria.
    from src.training.data_collator import _render_prefix

    prefix = _render_prefix("SP", "oi", history=None, environment={"os": "Windows"})
    assert "<environment>\nos: Windows\n</environment>" in prefix
    assert prefix.endswith("[ASSISTANT]\n")  # raw_text (treinado) começa DEPOIS do prefixo
