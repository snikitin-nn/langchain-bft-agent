"""Многоагентный конвейер генерации БФТ: генератор -> тестировщик -> критик.

Запуск:
    python pipeline.py --task "Добавить отказ от смены и подбор замены" --rules rules.example.txt
    python pipeline.py --task "..." --rules-text "Требования атомарны и измеримы; ..."

Архитектура (LangGraph):

    generate  ── LLM пишет черновик БФТ (по памятке и замечаниям критика)
    test      ── детерминированная проверка validate_bfr (12 правил структуры)
    critique  ── LLM-критик оценивает документ по ВАШИМ правилам (JSON-рецензия)
    decide    ── маршрутизация: готово или вернуться в generate (до N итераций)

Критик не дублирует проверку формата (её делает тестировщик), а оценивает
содержательно: полноту, однозначность, измеримость, покрытие ролей, логику.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from tools import get_analyst_memo, save_bfr, validate_bfr

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", encoding="utf-8")
load_dotenv(BASE_DIR.parent / ".env", encoding="utf-8")
load_dotenv(encoding="utf-8")

console = Console()

# ---------------------------------------------------------------------------
# Состояние конвейера и схемы рецензии
# ---------------------------------------------------------------------------

class Issue(BaseModel):
    """Одно замечание критика."""
    criterion: str = Field(description="Название нарушенного критерия/правила")
    severity: str = Field(description="blocker | major | minor")
    problem: str = Field(description="Что именно не так, желательно с местом")
    suggestion: str = Field(description="Как исправить")


class Review(BaseModel):
    """Структурированная рецензия критика."""
    approved: bool = Field(description="True — документ можно принимать, иначе False")
    summary: str = Field(description="Итоговая оценка в 1–2 предложениях")
    issues: list[Issue] = Field(default_factory=list, description="Список замечаний")


class State(TypedDict):
    task: str
    rules: str
    document: str
    tester_report: str
    review: dict
    feedback: str
    iterations: int


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

def _find_api_key() -> str | None:
    for name in ("DS_API_KEY", "DS_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name)
        if value:
            return value
    return None


def build_model() -> ChatOpenAI:
    api_key = _find_api_key()
    if not api_key:
        console.print("[red]Не найден ключ DeepSeek (DS_API_KEY / DS_KEY / DEEPSEEK_API_KEY).[/red]")
        raise SystemExit(2)
    return ChatOpenAI(
        model=os.getenv("DS_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("DS_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
        timeout=180,
        max_retries=2,
        max_tokens=8192,
    )


# ---------------------------------------------------------------------------
# Узлы графа
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM = """Ты — системный и бизнес-аналитик. Твоя задача — по описанию
функции написать ПОЛНЫЙ черновик БФТ для раздела «График» приложения НБЦ
(Нижегородская Библейская церковь).

Ниже — памятка аналитика (обязательные разделы, 12 правил, глоссарий, шаблон).
Следуй шаблону буквально: не сокращай структуру, воспроизводи ВСЕ разделы,
сценарии, приложения. Перед каждым разделом и подразделом ставь якорь
`.. _имя:` (как показано в шаблоне). Верни ТОЛЬКО текст документа в
reStructuredText, без пояснений и без вступлений.

ПАМЯТКА:
{memo}

Если в запросе есть «Замечания рецензента» — учти их и исправь документ."""


CRITIC_SYSTEM = """Ты — строгий рецензент-аналитик. Оцени черновик БФТ по
правилам ниже. Это ЕДИНСТВЕННЫЙ источник критериев — проверяй ровно по ним.

ПРАВИЛА:
{rules}

Оценивай содержательно: полноту требований, их однозначность и измеримость,
покрытие ролей, логику и полноту сценариев, наличие противоречий.
НЕ проверяй RST-разметку и структуру разделов — этим занимается другой агент.

Верни рецензию: approved=true только если документ полностью соответствует
правилам и можно принимать. Иначе approved=false и перечисли замечания
(не более 10 самых существенных) с критерием, severity (blocker/major/minor),
проблемой и кратким предложением по исправлению."""


def make_generate(model: ChatOpenAI, memo: str):
    system = GENERATOR_SYSTEM.format(memo=memo)

    def generate(state: State) -> dict:
        parts = [f"Описание функции:\n{state['task']}"]
        if state.get("feedback"):
            parts.append(f"Замечания рецензента (исправь их):\n{state['feedback']}")
        message = model.invoke([SystemMessage(system), HumanMessage("\n\n".join(parts))])
        content = message.content if isinstance(message.content, str) else str(message.content)
        return {
            "document": content,
            "iterations": (state.get("iterations") or 0) + 1,
        }

    return generate


def make_test():
    def test(state: State) -> dict:
        report = validate_bfr(state["document"])
        return {
            "tester_report": report,
            # в режиме без критика фидбек формирует сам тестировщик
            "feedback": report if "[FAIL]" in report else "",
        }

    return test


def make_critic(model: ChatOpenAI):
    # DeepSeek ненадёжен со structured output (json_schema / function_calling /
    # json_mode), поэтому просим вернуть JSON обычным текстом и парсим вручную
    # с ретраем и fallback'ом.
    json_instruction = (
        "\n\nФормат ответа — СТРОГО один JSON-объект, без markdown и пояснений:\n"
        '{"approved": <true|false>, "summary": "<1-2 предложения>", '
        '"issues": [{"criterion": "<критерий>", "severity": "<blocker|major|minor>", '
        '"problem": "<что не так>", "suggestion": "<как исправить>"}]}'
    )

    def _parse_json(text: str) -> dict | None:
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text.strip()).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
        return None

    def _fallback_review() -> dict:
        return {
            "approved": False,
            "summary": "Критик не смог сформировать структурированную рецензию (нет валидного JSON).",
            "issues": [{
                "criterion": "Формат рецензии",
                "severity": "major",
                "problem": "Модель не вернула валидный JSON.",
                "suggestion": "Повторить оценку документа.",
            }],
        }

    def critique(state: State) -> dict:
        console.print(f"[dim]⏳ Критик оценивает документ ({len(state['document'])} симв., ~1–2 мин)…[/dim]")
        messages = [
            SystemMessage(CRITIC_SYSTEM.format(rules=state["rules"]) + json_instruction),
            HumanMessage(f"Оцени документ:\n\n{state['document']}"),
        ]
        review_dict: dict | None = None
        for _ in range(2):
            try:
                resp = model.invoke(messages)
                text = resp.content if isinstance(resp.content, str) else str(resp.content)
                parsed = _parse_json(text)
                if parsed is not None:
                    Review.model_validate(parsed)
                    review_dict = parsed
                    break
            except Exception:
                review_dict = None
        if review_dict is None:
            review_dict = _fallback_review()

        # Собираем фидбек для генератора: замечания тестировщика + критика.
        feedback_parts: list[str] = []
        report = state.get("tester_report", "")
        if "[FAIL]" in report:
            feedback_parts.append(f"Замечания тестировщика (структура):\n{report}")
        if not review_dict.get("approved") and review_dict.get("issues"):
            lines = [
                f"- [{issue['severity']}] {issue['criterion']}: {issue['problem']} → {issue['suggestion']}"
                for issue in review_dict["issues"]
            ]
            feedback_parts.append("Замечания критика (содержание):\n" + "\n".join(lines))
        return {
            "review": review_dict,
            "feedback": "\n\n".join(feedback_parts) if feedback_parts else "",
        }

    return critique


def make_decide(max_iters: int, with_critic: bool):
    def decide(state: State) -> str:
        tester_ok = "[FAIL]" not in state.get("tester_report", "")
        critic_ok = True
        if with_critic:
            critic_ok = bool((state.get("review") or {}).get("approved"))
        iterations = state.get("iterations") or 0
        if (tester_ok and critic_ok) or iterations >= max_iters:
            return "finish"
        return "continue"

    return decide


def build_graph(model: ChatOpenAI, memo: str, max_iters: int, with_critic: bool):
    graph = StateGraph(State)
    graph.add_node("generate", make_generate(model, memo))
    graph.add_node("test", make_test())
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "test")
    if with_critic:
        graph.add_node("critique", make_critic(model))
        graph.add_edge("test", "critique")
        graph.add_conditional_edges("critique", make_decide(max_iters, True), {"continue": "generate", "finish": END})
    else:
        graph.add_conditional_edges("test", make_decide(max_iters, False), {"continue": "generate", "finish": END})
    return graph.compile()


# ---------------------------------------------------------------------------
# Запуск и вывод
# ---------------------------------------------------------------------------

def render_review(review: dict) -> None:
    approved = bool(review.get("approved"))
    title = "[bold green]Критик: ОДОБРЕНО[/bold green]" if approved else "[bold yellow]Критик: есть замечания[/bold yellow]"
    lines = [review.get("summary", "")]
    for issue in review.get("issues", []):
        lines.append(f"• [{issue['severity'].upper()}] {issue['criterion']}")
        lines.append(f"    Проблема: {issue['problem']}")
        lines.append(f"    Как чинить: {issue['suggestion']}")
    console.print(Panel(Text("\n".join(lines)), title=title, border_style="green" if approved else "yellow"))


def run_pipeline(task: str, rules: str | None, model: ChatOpenAI, memo: str, out: str, max_iters: int) -> None:
    with_critic = bool(rules)
    graph = build_graph(model, memo, max_iters, with_critic)
    console.print(Panel(Text(task), title="[bold]Задача[/bold]", border_style="white"))
    if with_critic:
        console.print(Panel(Text(rules), title="[bold]Правила критика[/bold]", border_style="white"))

    last_document = ""
    for chunk in graph.stream({"task": task, "rules": rules or ""}, stream_mode="updates"):
        for node, updates in chunk.items():
            if node == "generate":
                last_document = updates.get("document", "")
                console.print(f"[bold cyan]── Итерация {updates.get('iterations')}: генератор[/bold cyan] — документ {len(last_document)} симв.")
            elif node == "test":
                console.print(
                    Panel(Text(updates.get("tester_report", "")), title="[bold]Тестировщик (12 правил)[/bold]", border_style="magenta")
                )
            elif node == "critique":
                render_review(updates.get("review", {}))
        console.print(Rule(style="dim"))

    if last_document:
        result = save_bfr(last_document, out)
        console.print(Text(result, style="bold"))
    else:
        console.print("[red]Итоговый документ не получен.[/red]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Конвейер генерации БФТ: генератор → тестировщик → [критик]")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--task", help="описание функции")
    source.add_argument("--task-file", help="путь к файлу с описанием функции")
    rules = parser.add_mutually_exclusive_group()
    rules.add_argument("--rules", help="путь к файлу с правилами критика (без него критик отключён)")
    rules.add_argument("--rules-text", help="правила критика одной строкой")
    parser.add_argument("--out", default="nbc_bfr.rst", help="имя итогового файла в output/")
    parser.add_argument("--max-iters", type=int, default=3, help="максимум итераций цикла")
    return parser.parse_args(argv)


def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Файл не найден: {p}[/red]")
        raise SystemExit(2)
    return p.read_text(encoding="utf-8").strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    task = load_text(args.task_file) if args.task_file else (args.task or "").strip()
    if not task:
        console.print("[red]Укажите --task или --task-file[/red]")
        return 2
    rules: str | None = None
    if args.rules:
        rules = load_text(args.rules)
    elif args.rules_text:
        rules = args.rules_text.strip()

    model = build_model()
    memo = get_analyst_memo()
    run_pipeline(task, rules, model, memo, Path(args.out).name, args.max_iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
