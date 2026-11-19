"""Инструменты ИИ-агента, который готовит БФТ в формате Sphinx/reStructuredText.

Домен — приложение НБЦ (Нижегородская Библейская церковь): раздел «График»
со службами и сменами. Документ оформляется в Sphinx-стиле (заголовки `#`/`*`/`=`,
глоссарий через `.. glossary::`, сценарии со шагами, требования списками `#.`,
приложения с User Stories / OpenAPI / AS-IS·TO-BE / C4).

Модуль не зависит от LangChain: это чистые функции, тестируемые без модели.
`create_agent` принимает обычные функции с докстрингами и сам строит схемы.

Публичные функции (инструменты агента):
    get_analyst_memo()          — памятка: 12 правил, глоссарий, синонимы, шаблон
    validate_bfr(text)          — проверка документа по 12 правилам
    save_bfr(text, filename)    — сохранить документ в output/
    read_bfr(filename)          — прочитать ранее сохранённый документ
    make_unified_diff(a, b)     — патч между двумя версиями документа
"""

from __future__ import annotations

import difflib
import re
from datetime import date
from pathlib import Path
from typing import Callable

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# ---------------------------------------------------------------------------
# Домен: приложение НБЦ
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS: tuple[str, ...] = (
    "Термины и определения",
    "Введение",
    "Общие сведения",
    "Сценарии использования",
    "Функциональные требования",
    "Нефункциональные требования",
    "Приложения",
)

#: Глоссарий домена: термин -> определение.
GLOSSARY: dict[str, str] = {
    "График": "план служений и смен церкви на определённый период",
    "Смена": "период служения, закреплённый за Участником в рамках графика",
    "Служение": "направление деятельности церкви, для которого формируется график",
    "Участник": "пользователь Приложения, включённый в график служений",
    "Ответственный": "участник, которому Координатор делегировал управление служением",
    "Координатор": "участник, отвечающий за несколько служений и назначающий Ответственных",
    "Задача служения": "выделенная задача в рамках служения, на которую назначается Участник",
    "Подтверждение": "действие Участника, которым он подтверждает участие в смене",
    "Вакантна": "статус смены без назначенного Участника",
    "Журнал событий графика": "записи о действиях участников и изменениях графика",
    "Система": "раздел «График» в составе Приложения",
    "Приложение": "веб-приложение (PWA) церкви НБЦ",
}

#: Запрещённые синонимы -> канонический термин глоссария.
FORBIDDEN_TERMS: dict[str, str] = {
    "расписание": "График",
    "дежурство": "Смена",
    "волонтёр": "Участник",
    "куратор": "Ответственный",
    "юзер": "Участник",
    "дедлайн": "срок",
    "фича": "функция",
}

#: Ключевые термины, которые в тексте должны быть выделены полужирным `**Термин**`.
#: «Приложение» сознательно исключено: оно совпадает с заголовками «Приложение N.»,
#: которые по RST не выделяются.
KEY_TERMS: tuple[str, ...] = ("Система", "Участник", "Ответственный", "Координатор")

PLACEHOLDER_RE = re.compile(r"\b(TBD|TODO|XXX|FIXME)\b|«\.\.\.»|<[^<>\n]{2,40}>")


# ---------------------------------------------------------------------------
# Разбор RST/Sphinx: заголовки и тела разделов
# ---------------------------------------------------------------------------

_HEADING_UNDERLINE_RE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")

#: Порядок уровней заголовков (меньше = выше в иерархии). Совпадает с Sphinx.
_LEVEL: dict[str, int] = {"#": 0, "*": 1, "=": 2, "-": 3, "^": 4, "~": 5, '"': 6, "'": 7}


def _headings(text: str) -> list[tuple[str, str, int]]:
    """Все заголовки RST: [(заголовок, символ_уровня, индекс_строки_подчёркивания)].

    Нумерация строк 0-based, как в splitlines().
    """
    lines = text.splitlines()
    headings: list[tuple[str, str, int]] = []
    for i in range(1, len(lines)):
        match = _HEADING_UNDERLINE_RE.match(lines[i])
        if match and lines[i - 1].strip():
            headings.append((lines[i - 1].strip(), match.group(1), i))
    return headings


def _section_body(text: str, needle: str) -> str | None:
    """Тело первого раздела, в заголовке которого встречается `needle`.

    Тело тянется до следующего заголовка того же или более высокого уровня,
    поэтому вложенные подразделы считаются частью раздела.
    """
    lines = text.splitlines()
    headings = _headings(text)
    for n, (title, char, underline) in enumerate(headings):
        if needle.lower() in title.lower():
            level = _LEVEL.get(char, 0)
            end = len(lines)
            for _next_title, next_char, next_underline in headings[n + 1:]:
                if _LEVEL.get(next_char, 0) <= level:
                    end = next_underline - 1
                    break
            return "\n".join(lines[underline + 1:end])
    return None


def _numbered_lines(body: str) -> list[str]:
    """Пункты нумерованного (`#.`, `1.`, `1)`) или маркированного списка."""
    return [line for line in body.splitlines() if re.match(r"^\s*(#\.|\d+[.)]|[-*+])\s+\S", line)]


# ---------------------------------------------------------------------------
# 12 правил памятки: (код, название, проверка) -> список замечаний
# ---------------------------------------------------------------------------

def _check_formatting(text: str) -> list[str]:
    """R01. Документ начинается с заголовка; строки до 120 символов; без табов."""
    lines = text.splitlines()
    if not _headings(text):
        return ["нет ни одного RST-заголовка (строка + подчёркивание из #, *, =)"]

    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return ["документ пуст"]

    has_title = False
    if first + 1 < len(lines) and _HEADING_UNDERLINE_RE.match(lines[first + 1]):
        has_title = True
    elif first + 2 < len(lines) and _HEADING_UNDERLINE_RE.match(lines[first]) and _HEADING_UNDERLINE_RE.match(lines[first + 2]):
        has_title = True

    problems: list[str] = []
    if not has_title:
        problems.append("документ не начинается с заголовка")

    for number, line in enumerate(lines, 1):
        if len(line) > 120:
            problems.append(f"строка {number}: длина {len(line)} символов при лимите 120")
        if "\t" in line:
            problems.append(f"строка {number}: символ табуляции")
    return problems[:10]


def _check_required_sections(text: str) -> list[str]:
    """R02. Все обязательные разделы присутствуют и не пусты."""
    problems: list[str] = []
    for name in REQUIRED_SECTIONS:
        body = _section_body(text, name)
        if body is None:
            problems.append(f"нет раздела «{name}»")
            continue
        content = [
            line for line in body.splitlines()
            if line.strip() and not line.lstrip().startswith("..")
        ]
        if not content:
            problems.append(f"раздел «{name}» пуст")
    return problems


def _check_glossary(text: str) -> list[str]:
    """R03. Глоссарий оформлен директивой `.. glossary::` с минимум 6 терминами."""
    body = _section_body(text, "Термины и определения")
    if body is None:
        return ["нет раздела «Термины и определения»"]
    if ".. glossary::" not in body:
        return ["глоссарий должен быть оформлен директивой `.. glossary::`"]
    terms = re.findall(r"^\s*:([^:]+):", body, re.MULTILINE)
    if len(terms) < 6:
        return [f"в глоссарии {len(terms)} терминов, нужно минимум 6 в формате `:Термин: определение`"]
    return []


def _check_placeholders(text: str) -> list[str]:
    """R04. Нет заглушек (TBD, TODO, «...», <...>)."""
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        for match in PLACEHOLDER_RE.finditer(line):
            problems.append(f"строка {number}: заглушка {match.group(0)!r}")
    return problems[:10]


def _check_bold_terms(text: str) -> list[str]:
    """R05. Ключевые термины глоссария выделены полужирным `**Термин**`."""
    problems: list[str] = []
    for term in KEY_TERMS:
        if re.search(rf"\*\*{re.escape(term)}\*\*", text):
            continue
        if re.search(rf"\b{re.escape(term)}\b", text):
            problems.append(f"термин «{term}» употребляется без полужирного выделения `**{term}**`")
    return problems[:8]


def _check_scenarios(text: str) -> list[str]:
    """R06. Сценарии пронумерованы «Сценарий N. …» (минимум 3) с «Кратким описанием»."""
    body = _section_body(text, "Сценарии использования")
    if body is None:
        return ["нет раздела «Сценарии использования»"]
    scenarios = re.findall(r"^Сценарий\s+\d+\s*\.\s", body, re.MULTILINE)
    if len(scenarios) < 3:
        return [f"сценариев «Сценарий N. …»: {len(scenarios)}, нужно минимум 3"]
    descriptions = len(re.findall(r"Краткое описание", body))
    if descriptions < len(scenarios):
        return [f"у {len(scenarios)} сценариев только {descriptions} блоков «Краткое описание»"]
    return []


def _check_scenario_steps(text: str) -> list[str]:
    """R07. У сценариев есть шаги: таблица `.. csv-table::` или нумерованный список."""
    body = _section_body(text, "Сценарии использования")
    if body is None:
        return ["нет раздела «Сценарии использования»"]
    tables = len(re.findall(r"\.\.\s+csv-table::", body))
    numbered = len(re.findall(r"^\s*(\d+[.)]|#\.)\s+\S", body, re.MULTILINE))
    if tables + numbered < 3:
        return [f"шагов в сценариях: {tables} таблиц + {numbered} нумерованных строк, нужно минимум 3"]
    return []


def _check_functional_requirements(text: str) -> list[str]:
    """R08. Функциональные требования оформлены нумерованным списком `#.` (минимум 3)."""
    body = _section_body(text, "Функциональные требования")
    if body is None:
        return ["нет раздела «Функциональные требования»"]
    items = re.findall(r"^\s*#\.\s+\S", body, re.MULTILINE)
    if len(items) < 3:
        return [f"требований в списке «#.»: {len(items)}, нужно минимум 3"]
    return []


def _check_nonfunctional(text: str) -> list[str]:
    """R09. Описаны нефункциональные требования (минимум 2 пункта)."""
    body = _section_body(text, "Нефункциональные требования")
    if body is None:
        return ["нет раздела «Нефункциональные требования»"]
    items = _numbered_lines(body)
    if len(items) < 2:
        return [f"нефункциональных требований: {len(items)}, нужно минимум 2"]
    return []


def _check_reporting(text: str) -> list[str]:
    """R10. Описаны журнал событий и метрики мониторинга."""
    body = _section_body(text, "Отчётность")
    if body is None:
        return ["нет раздела «Отчётность и метрики»"]
    problems: list[str] = []
    if "Журнал" not in body:
        problems.append("не описан журнал событий графика")
    metrics = _numbered_lines(body)
    if len(metrics) < 2:
        problems.append(f"метрик мониторинга: {len(metrics)}, нужно минимум 2")
    return problems


def _check_terminology(text: str) -> list[str]:
    """R11. Единая терминология по глоссарию (нет запрещённых синонимов)."""
    problems: list[str] = []
    for term, canonical in FORBIDDEN_TERMS.items():
        if re.search(rf"\b{re.escape(term)}\w*", text, re.IGNORECASE):
            problems.append(f"использован синоним «{term}» — в глоссарии принят термин «{canonical}»")
    return problems


def _check_appendix(text: str) -> list[str]:
    """R12. Объявлены приложения (User Stories, OpenAPI, AS-IS/TO-BE, C4)."""
    body = _section_body(text, "Приложения")
    if body is None:
        return ["нет раздела «Приложения»"]
    markers = ("User Stor", "OpenAPI", "AS-IS", "TO-BE", "C4", "BPMN")
    found = [marker for marker in markers if marker in body]
    if len(found) < 2:
        return [f"в приложениях объявлено {len(found)} артефактов, нужно минимум 2 (User Stories, OpenAPI, AS-IS/TO-BE, C4)"]
    return []


RULES: list[tuple[str, str, Callable[[str], list[str]]]] = [
    ("R01", "Заголовок документа и форматирование (строки ≤ 120, без табов)", _check_formatting),
    ("R02", "Наличие и непустота обязательных разделов", _check_required_sections),
    ("R03", "Глоссарий `.. glossary::` с определениями", _check_glossary),
    ("R04", "Нет заглушек (TBD, TODO, «...», <...>)", _check_placeholders),
    ("R05", "Ключевые термины выделены полужирным `**`", _check_bold_terms),
    ("R06", "Сценарии пронумерованы и имеют «Краткое описание»", _check_scenarios),
    ("R07", "У сценариев есть шаги (таблица или список)", _check_scenario_steps),
    ("R08", "Функциональные требования списком `#.`", _check_functional_requirements),
    ("R09", "Нефункциональные требования описаны", _check_nonfunctional),
    ("R10", "Отчётность: журнал событий и метрики", _check_reporting),
    ("R11", "Единая терминология по глоссарию", _check_terminology),
    ("R12", "Приложения объявлены (User Stories / OpenAPI / AS-IS·TO-BE / C4)", _check_appendix),
]


# ---------------------------------------------------------------------------
# Инструменты агента
# ---------------------------------------------------------------------------

def get_analyst_memo() -> str:
    """Вернуть памятку аналитика НБЦ: 12 правил, глоссарий и запрещённые синонимы.

    Вызывай первым делом — памятка задаёт требования, по которым документ
    будет проверяться инструментом validate_bfr.
    """
    lines = [
        "ПАМЯТКА АНАЛИТИКА: БФТ ПРИЛОЖЕНИЯ НБЦ (раздел «График»)",
        "=" * 56,
        "",
        "Документ оформляется в Sphinx/reStructuredText. Иерархия заголовков:",
        "  # — разделы первого уровня (Термины и определения, Введение, …)",
        "  * — подразделы",
        "  = — под-подразделы (сценарии, приложения)",
        "",
        "Обязательные разделы:",
        *[f"  - {name}" for name in REQUIRED_SECTIONS],
        "  - Отчётность и метрики",
        "",
        "12 правил проверки:",
        *[f"  {code}. {title}" for code, title, _ in RULES],
        "",
        "Глоссарий (термины обязательны к использованию и выделяются **полужирным**):",
        *[f"  - {term} — {definition}" for term, definition in GLOSSARY.items()],
        "",
        "Запрещённые синонимы (используйте канонический термин):",
        *[f"  - «{term}» -> «{canonical}»" for term, canonical in FORBIDDEN_TERMS.items()],
        "",
        "Шаблон документа:",
        "  Термины и определения",
        "  #####################",
        "  .. glossary::",
        "     :Термин: определение",
        "",
        "  Введение",
        "  ########",
        "  Аннотация",
        "  *********",
        "  ...",
        "",
        "  Общие сведения",
        "  ##############",
        "  ...",
        "",
        "  Сценарии использования",
        "  **********************",
        "  Сценарий 1. Название",
        "  =====================",
        "  Сценарий обеспечивает ...",
        "  Краткое описание сценария: ...",
        "  .. csv-table:: Основные шаги сценария 1",
        "     :header: \"Кто\", \"Действие\"",
        "     \"Роль\", \"Действие\"",
        "  Альтернативные сценарии:",
        "  * ...",
        "",
        "  Функциональные требования",
        "  #########################",
        "  #. Система должна ...",
        "",
        "  Отчётность и метрики",
        "  ####################",
        "  Журнал событий",
        "  **************",
        "  Метрики",
        "  *******",
        "  #. ...",
        "",
        "  Нефункциональные требования",
        "  ###########################",
        "  #. ...",
        "",
        "  Приложения",
        "  ##########",
        "  Приложение 1. AS-IS / TO-BE",
        "  ***************************",
        "  Приложение 2. User Stories",
        "  **************************",
        "  Приложение 3. OpenAPI",
        "  *********************",
    ]
    return "\n".join(lines)


def validate_bfr(text: str) -> str:
    """Проверить документ БФТ НБЦ по 12 правилам и вернуть текстовый отчёт.

    Передавай полный текст документа целиком. Вызывай после генерации черновика
    и после каждого исправления.
    """
    if not text or not text.strip():
        return "ОШИБКА: передан пустой документ, проверять нечего."

    header = [
        f"Отчёт валидации БФТ НБЦ от {date.today():%d.%m.%Y}",
        "=" * 44,
    ]
    body: list[str] = []
    passed = 0

    for code, title, checker in RULES:
        try:
            problems = checker(text)
        except Exception as exc:  # проверка не должна ронять агента
            problems = [f"внутренняя ошибка проверки: {exc}"]
        if problems:
            body.append(f"[FAIL] {code} {title}")
            body.extend(f"       - {problem}" for problem in problems[:8])
        else:
            passed += 1
            body.append(f"[ OK ] {code} {title}")

    footer = [
        "-" * 44,
        f"Пройдено {passed} из {len(RULES)} правил.",
    ]
    if passed == len(RULES):
        footer.append("Документ соответствует памятке — сохрани его через save_bfr.")
    else:
        footer.append("Документ НЕ соответствует памятке. Исправь замечания и вызови validate_bfr снова.")

    return "\n".join(header + body + footer)


def save_bfr(text: str, filename: str = "nbc_bfr.rst") -> str:
    """Сохранить готовый документ БФТ в файл output/<filename>.

    Вызывай только когда validate_bfr не вернул ни одного [FAIL].
    Возвращает путь к сохранённому файлу.
    """
    if not text or len(text.strip()) < 200:
        return "ОШИБКА: документ пуст или слишком короткий (меньше 200 символов), не сохраняю."

    safe_name = Path(filename).name or "nbc_bfr.rst"
    if not safe_name.lower().endswith((".rst", ".txt")):
        safe_name += ".rst"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / safe_name
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return f"Сохранено {len(text)} символов в {path}"


def read_bfr(filename: str = "nbc_bfr.rst") -> str:
    """Прочитать ранее сохранённый черновик из output/<filename>."""
    path = OUTPUT_DIR / (Path(filename).name or "nbc_bfr.rst")
    if not path.exists():
        return f"ОШИБКА: файл не найден: {path}"
    return path.read_text(encoding="utf-8")


def make_unified_diff(
    original: str,
    revised: str,
    fromfile: str = "draft_v1.rst",
    tofile: str = "draft_v2.rst",
) -> str:
    """Построить unified diff между двумя версиями документа."""
    diff = difflib.unified_diff(
        original.splitlines(),
        revised.splitlines(),
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
        n=3,
    )
    patch = "\n".join(diff)
    return patch if patch.strip() else "Изменений нет."
