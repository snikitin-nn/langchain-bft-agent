"""Быстрый режим БФТ-генератора: генератор + детерминированный тестировщик.

Без агента-критика. Полный конвейер с критиком — в pipeline.py:

    python pipeline.py --task "..." --rules rules.example.txt

Запуск:
    python main.py --task "Добавить отказ от смены «Не могу прийти» и подбор замены"
    python main.py --task-file task.txt --out nbc_bfr.rst
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import build_model, run_pipeline
from tools import get_analyst_memo

DEFAULT_TASK = "Добавить отказ от смены «Не могу прийти» и подбор замены"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генератор + тестировщик БФТ (без критика)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--task", help="описание функции")
    source.add_argument("--task-file", help="путь к файлу с описанием функции")
    parser.add_argument("--out", default="nbc_bfr.rst", help="имя итогового файла в output/")
    parser.add_argument("--max-iters", type=int, default=3, help="максимум итераций генерации/проверки")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.task_file:
        path = Path(args.task_file)
        if not path.exists():
            print(f"Файл не найден: {path}")
            return 2
        task = path.read_text(encoding="utf-8").strip()
    else:
        task = (args.task or DEFAULT_TASK).strip()

    model = build_model()
    memo = get_analyst_memo()
    run_pipeline(task, rules=None, model=model, memo=memo, out=Path(args.out).name, max_iters=args.max_iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
