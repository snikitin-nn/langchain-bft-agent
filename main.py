"""ИИ-агент для подготовки черновиков БФТ приложения НБЦ (LangChain 1.x + DeepSeek).

Запуск:
    python main.py                                   # демо-задача из README
    python main.py --task "Добавить отказ от смены и подбор замены"
    python main.py --task-file task.txt --out nbc_bfr.rst

Агент работает по циклу «памятка -> черновик -> валидация -> патч -> сохранение»
и использует инструменты из tools.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from tools import (
    OUTPUT_DIR,
    get_analyst_memo,
    make_unified_diff,
    read_bfr,
    save_bfr,
    validate_bfr,
)

BASE_DIR = Path(__file__).resolve().parent

# Читаем .env из трёх мест (load_dotenv не перетирает уже заданные значения):
#  1) .env рядом с проектом (DEEPSEEK_API_KEY и т.п.)
#  2) .env рабочей папки C:\DSH_Sandbox (DS_KEY, DS_BASE_URL)
#  3) .env в текущем каталоге запуска
# encoding="utf-8" обязателен: иначе на Windows с русской локалью python-dotenv
# читает .env в cp1251 и падает на кириллических комментариях.
load_dotenv(BASE_DIR / ".env", encoding="utf-8")
load_dotenv(BASE_DIR.parent / ".env", encoding="utf-8")
load_dotenv(encoding="utf-8")

console = Console()

DEFAULT_TASK = (
    "Добавить в раздел «График» приложения НБЦ возможность участнику отметить "
    "«Не могу прийти» по смене, а системе — предложить замену из участников "
    "служения и уведомить ответственного."
)

SYSTEM_PROMPT = """Ты — системный и бизнес-аналитик. Твоя задача — по описанию
функции подготовить черновик БФТ (бизнес-функциональных требований) для раздела
«График» приложения НБЦ (Нижегородская Библейская церковь).

Документ оформляется в Sphinx/reStructuredText. Иерархия заголовков:
'#' — разделы, '*' — подразделы, '=' — под-подразделы.

Строгий порядок работы, не пропускай шаги:
1. Вызови инструмент get_analyst_memo — это памятка из 12 правил, глоссарий
   и список запрещённых синонимов. Дальше следуй ей буквально.
2. Напиши черновик БФТ с разделами: Термины и определения (директива
   .. glossary::), Введение, Общие сведения, Сценарии использования,
   Функциональные требования, Отчётность и метрики, Нефункциональные
   требования, Приложения (AS-IS/TO-BE, User Stories, OpenAPI, C4).
3. Вызови validate_bfr и передай в него ПОЛНЫЙ текст черновика.
4. Если в отчёте есть [FAIL] — исправь все замечания, затем снова вызови
   validate_bfr. Сделай не больше трёх итераций.
5. Когда [FAIL] не осталось, вызови save_bfr с полным текстом документа.
6. В финальном ответе верни ТОЛЬКО текст RST-документа, без пояснений
   и без вступлений вроде «вот ваш документ».

Правила оформления:
- Глоссарий — через директиву `.. glossary::`, каждый термин строкой
  `:Термин: определение`. Минимум 6 терминов.
- Ключевые термины глоссария в тексте выделяй полужирным: **Система**,
  **Приложение**, **Участник**, **Ответственный**, **Координатор**, **Смена**.
- Сценарии нумеруй «Сценарий 1. Название» (минимум 3). У каждого сценария:
  «Сценарий обеспечивает …», «Краткое описание сценария: …», шаги в таблице
  `.. csv-table::` (или нумерованным списком) и «Альтернативные сценарии:».
- Функциональные и нефункциональные требования — нумерованным списком `#.`.
- В «Отчётность и метрики» опиши журнал событий и минимум 2 метрики.
- В «Приложения» объяви хотя бы два артефакта: User Stories, OpenAPI,
  AS-IS/TO-BE (BPMN) или C4.
- Не оставляй заглушек: TBD, TODO, «...», <вставьте здесь>.
- Пиши по-русски; латиница допустима только для API, REST, HTTP, JSON, RST,
  OpenAPI, User Stories, AS-IS, TO-BE, BPMN, C4, Gherkin.
- Если данных не хватает — сделай разумное допущение и отрази его в тексте.
  Вопросов не задавай, работай автономно.
"""

RECURSION_LIMIT = 60

#: Инструменты агента. create_agent принимает обычные функции с докстрингами.
TOOLS = [get_analyst_memo, validate_bfr, save_bfr, read_bfr, make_unified_diff]


def _find_api_key() -> str | None:
    """Найти ключ DeepSeek под любым из распространённых имён переменных."""
    for name in ("DS_API_KEY", "DS_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name)
        if value:
            return value
    return None


def build_model() -> ChatOpenAI:
    """Собрать модель DeepSeek через OpenAI-совместимый интерфейс."""
    api_key = _find_api_key()
    if not api_key:
        console.print(
            Panel(
                "Не найден DS_API_KEY.\n\n"
                "Создайте файл .env рядом с main.py (см. .env.example):\n"
                "  DS_API_KEY=sk-...\n"
                "  DS_BASE_URL=https://api.deepseek.com\n"
                "  DS_MODEL=deepseek-chat",
                title="[red]Ошибка конфигурации[/red]",
                border_style="red",
            )
        )
        raise SystemExit(2)

    return ChatOpenAI(
        model=os.getenv("DS_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("DS_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
        timeout=180,
        max_retries=2,
    )


def _content_to_text(content: object) -> str:
    """Привести content сообщения (строка или список блоков) к тексту."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def render_message(message: BaseMessage) -> None:
    """Напечатать одно сообщение агента человекочитаемо.

    Весь динамический текст оборачивается в Text: иначе Rich примет
    квадратные скобки из RST и отчёта ([ OK ], [FAIL]) за разметку и упадёт.
    """
    if isinstance(message, AIMessage):
        text = _content_to_text(message.content).strip()
        if text:
            console.print(
                Panel(Text(text), title="[bold]Ассистент[/bold]", border_style="cyan", expand=False)
            )
        for call in message.tool_calls or []:
            line = Text("  -> ", style="bold green")
            line.append(str(call.get("name")), style="bold green")
            line.append(f" {call.get('args')}", style="dim")
            console.print(line)
    elif isinstance(message, ToolMessage):
        answer = _content_to_text(message.content).strip()
        preview = answer if len(answer) <= 700 else answer[:700] + "\n… (сокращено)"
        console.print(
            Panel(
                Text(preview),
                title=f"[bold]Результат: {escape(str(message.name))}[/bold]",
                border_style="magenta",
            )
        )
    else:
        console.print(Text(f"{type(message).__name__}: {_content_to_text(message.content)[:300]}"), style="dim")


def run_agent(task: str) -> list[BaseMessage]:
    """Прогнать агента по задаче и вернуть финальный список сообщений."""
    model = build_model()
    agent = create_agent(model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)

    payload = {"messages": [{"role": "user", "content": f"Описание задачи из трекера:\n\n{task}"}]}
    config = {"recursion_limit": RECURSION_LIMIT}

    messages: list[BaseMessage] = []
    printed = 0
    for chunk in agent.stream(payload, stream_mode="values", config=config):
        messages = chunk["messages"]
        for message in messages[printed:]:
            render_message(message)
        printed = len(messages)
        console.print(Rule(style="dim"))

    return messages


def extract_final_document(messages: list[BaseMessage]) -> str:
    """Найти последний ответ ассистента без вызовов инструментов."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = _content_to_text(message.content).strip()
            if text:
                return text
    return ""


def finalize(final_text: str, out_name: str) -> None:
    """Проверить итоговый документ, сохранить и показать сводку."""
    document = final_text
    if document:
        document = document.strip()
        if document.startswith("```"):
            lines = document.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            document = "\n".join(lines).strip()

    saved_path = OUTPUT_DIR / out_name
    if document and len(document) > 200:
        save_bfr(document, out_name)
    elif saved_path.exists():
        document = read_bfr(out_name)
    else:
        console.print("[red]Агент не вернул документ и ничего не сохранил.[/red]")
        return

    report = validate_bfr(document)
    ok = "[FAIL]" not in report
    console.print(
        Panel(
            Text(report),
            title="[bold]Итоговая валидация[/bold]",
            border_style="green" if ok else "yellow",
            expand=False,
        )
    )
    console.print(Text(f"Файл: {saved_path}", style="bold"))

    previous = OUTPUT_DIR / "bfr_previous.rst"
    if previous.exists():
        diff = make_unified_diff(
            previous.read_text(encoding="utf-8"),
            document,
            fromfile="bfr_previous.rst",
            tofile=out_name,
        )
        if diff != "Изменений нет.":
            console.print(
                Panel(Text(diff), title="[bold]Патч к предыдущей версии[/bold]", border_style="blue")
            )
    previous.write_text(document, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ИИ-агент для генерации черновиков БФТ")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--task", help="описание задачи из трекера")
    source.add_argument("--task-file", help="путь к файлу с описанием задачи")
    parser.add_argument(
        "--out",
        default="bfr_draft.rst",
        help="имя итогового файла в output/ (по умолчанию bfr_draft.rst)",
    )
    return parser.parse_args(argv)


def read_task(args: argparse.Namespace) -> str:
    if args.task_file:
        path = Path(args.task_file)
        if not path.exists():
            console.print(f"[red]Файл с задачей не найден:[/red] {escape(str(path))}")
            raise SystemExit(2)
        return path.read_text(encoding="utf-8").strip()
    return (args.task or DEFAULT_TASK).strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    task = read_task(args)

    console.print(
        Panel(Text(task), title="[bold]Задача из трекера[/bold]", border_style="white", expand=False)
    )

    messages = run_agent(task)
    finalize(extract_final_document(messages), Path(args.out).name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
