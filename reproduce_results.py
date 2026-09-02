#!/usr/bin/env python3
"""Run the published monoterpene synthase model analyses."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


@dataclass(frozen=True)
class Workflow:
    description: str
    notebooks: tuple[str, ...]
    datasets: tuple[str, ...]
    packages: tuple[str, ...] = ()
    optional: bool = False


WORKFLOWS = {
    "test_train": Workflow(
        description="Train the final model and evaluate the independent holdout set.",
        notebooks=("TestTrainEval.ipynb", "analysis/TestTrainEval.ipynb"),
        datasets=("TestTrainSet.csv", "ValidationSet.csv"),
    ),
    "loo": Workflow(
        description="Run leave-one-out validation on the training partition.",
        notebooks=("LOOXGBoost.ipynb", "analysis/LOOXGBoost.ipynb"),
        datasets=("FullDataset.csv",),
    ),
    "shap": Workflow(
        description=(
            "Run repeated stratified five-fold validation and generate SHAP, "
            "gain, and confusion-matrix figures."
        ),
        notebooks=(
            "SHAP_and_Gain_Analysis.ipynb",
            "analysis/SHAP_and_Gain_Analysis.ipynb",
        ),
        datasets=("FullDataset.csv",),
        packages=("shap",),
    ),
    "pluskal": Workflow(
        description="Evaluate the final model on the external Pluskal dataset.",
        notebooks=(
            "TestOnPluskalDataset.ipynb",
            "analysis/TestOnPluskalDataset.ipynb",
            "SupplementaryInformation/TestOnPluskalDataset.ipynb",
        ),
        datasets=("FullDataset.csv", "PreprintExtraLinLimData.csv"),
    ),
    "loco": Workflow(
        description="Run sequence-, structure-, and all-feature-based LOCO validation.",
        notebooks=(
            "SupplementaryInformation/LOCO/LOCO_Validation.ipynb",
            "LOCO_Validation.ipynb",
        ),
        datasets=("FullDataset.csv",),
        optional=True,
    ),
}

MAIN_WORKFLOWS = ("test_train", "loo", "shap", "pluskal")

LOCO_GENERATORS = (
    "SupplementaryInformation/LOCO/Generate_LOCO_Splits.ipynb",
    "Generate_LOCO_Splits.ipynb",
)

DEPENDENCIES = {
    "nbconvert": "nbconvert",
    "ipykernel": "ipykernel",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
}

INSTALL_LINE = re.compile(
    r"^\s*(?:!(?:\{[^}]+\}\s+-m\s+)?|%)pip(?:3)?\s+install\b.*$",
    flags=re.MULTILINE,
)

PLUSKAL_TRAINING_FILE = re.compile(
    r"^(\s*TRAIN_CSV\s*=\s*)(['\"])Dataset\.csv\2",
    flags=re.MULTILINE,
)

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

RESULT_TITLES = {
    "test_train": "Independent holdout",
    "loo": "Leave-one-out validation",
    "shap": "Repeated stratified five-fold validation",
    "pluskal": "External Pluskal validation",
    "loco": "Leave-one-cluster-out validation",
}


def repository_root() -> Path:
    return Path(__file__).resolve().parent


def find_notebook(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for relative_path in candidates:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the fixed-feature XGBoost model evaluations without "
            "repeating Bayesian feature or hyperparameter optimization."
        )
    )
    parser.add_argument(
        "--analysis",
        nargs="+",
        choices=("all", "model", *WORKFLOWS),
        default=["all"],
        help=(
            "Analyses to run. The default runs the four main workflows and "
            "includes LOCO when its published notebook is present."
        ),
    )
    parser.add_argument(
        "--regenerate-loco-splits",
        action="store_true",
        help="Regenerate LOCO splits before evaluation; requires MMseqs2 and Foldseek.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="Per-cell notebook timeout in seconds; -1 disables the timeout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check notebook and dataset paths without running the analyses.",
    )
    arguments = parser.parse_args()
    if arguments.timeout < -1:
        parser.error("--timeout must be -1 or a non-negative number of seconds.")
    return arguments


def selected_workflows(root: Path, requested: list[str]) -> list[str]:
    if "all" in requested:
        selected = list(MAIN_WORKFLOWS)
        if find_notebook(root, WORKFLOWS["loco"].notebooks) is not None:
            selected.append("loco")
        return selected

    expanded: list[str] = []
    for name in requested:
        choices = MAIN_WORKFLOWS if name == "model" else (name,)
        for choice in choices:
            if choice not in expanded:
                expanded.append(choice)
    return expanded


def check_dependencies(selected: list[str]) -> None:
    modules = dict(DEPENDENCIES)
    for workflow_name in selected:
        for module in WORKFLOWS[workflow_name].packages:
            modules[module] = module

    missing = [package for module, package in modules.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        packages = " ".join(missing)
        raise ModuleNotFoundError(
            "Required Python packages are missing: "
            f"{', '.join(missing)}\n"
            "Activate the repository environment or install them with:\n"
            f"  {sys.executable} -m pip install {packages}"
        )


def check_datasets(root: Path, selected: list[str]) -> None:
    required = dict.fromkeys(
        filename
        for workflow_name in selected
        for filename in WORKFLOWS[workflow_name].datasets
    )
    missing = [filename for filename in required if not (root / filename).is_file()]
    if missing:
        details = "\n".join(f"- {filename}" for filename in missing)
        raise FileNotFoundError(
            f"Required datasets are missing from the repository root:\n{details}"
        )


def prepare_notebook(notebook: Path, workflow_name: str, root: Path) -> Path:
    document = json.loads(notebook.read_text(encoding="utf-8"))
    pluskal_assignment_updated = False

    for cell in document.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        original = cell.get("source", "")
        source = "".join(original) if isinstance(original, list) else original
        source = INSTALL_LINE.sub(
            "# Dependencies are installed in the repository environment.", source
        )

        if workflow_name == "pluskal":
            relative_dataset = Path(
                os.path.relpath(root / "FullDataset.csv", notebook.parent)
            ).as_posix()
            source, replacements = PLUSKAL_TRAINING_FILE.subn(
                lambda match: match.group(1) + repr(relative_dataset), source
            )
            pluskal_assignment_updated |= bool(replacements)

        cell["source"] = source.splitlines(keepends=True)

    if workflow_name == "pluskal" and not pluskal_assignment_updated:
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in document.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        if "FullDataset.csv" not in code:
            raise ValueError(
                "The Pluskal notebook does not identify FullDataset.csv as its "
                "training dataset and its existing TRAIN_CSV assignment could "
                "not be updated automatically."
            )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{notebook.stem}.reproduction.",
        suffix=".ipynb",
        dir=notebook.parent,
        delete=False,
    ) as temporary_file:
        json.dump(document, temporary_file, ensure_ascii=False, indent=1)
        temporary_file.write("\n")
        return Path(temporary_file.name)


def execute_notebook(
    notebook: Path,
    workflow_name: str,
    root: Path,
    output_directory: Path,
    timeout: int,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    output_name = f"{notebook.stem}.executed.ipynb"
    prepared = prepare_notebook(notebook, workflow_name, root)

    command = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(prepared),
        "--output",
        output_name,
        "--output-dir",
        str(output_directory),
        f"--ExecutePreprocessor.timeout={timeout}",
    ]
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("PYTHONUNBUFFERED", "1")

    print(f"\nRunning {notebook.relative_to(root)}", flush=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=notebook.parent, env=environment)
        if completed.returncode:
            raise RuntimeError(
                f"{workflow_name} failed with exit code {completed.returncode}: "
                f"{notebook.relative_to(root)}"
            )
    finally:
        prepared.unlink(missing_ok=True)

    executed = output_directory / output_name
    if not executed.is_file():
        raise FileNotFoundError(
            f"Notebook execution finished without producing {executed}."
        )

    elapsed = time.monotonic() - started
    print(f"Completed {workflow_name} in {elapsed:.1f} seconds.", flush=True)
    return executed


def notebook_text_output(notebook: Path) -> str:
    """Collect plain-text outputs from an executed notebook."""
    document = json.loads(notebook.read_text(encoding="utf-8"))
    text_outputs: list[str] = []

    for cell in document.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                value = output.get("text", "")
            else:
                value = output.get("data", {}).get("text/plain", "")

            if isinstance(value, list):
                value = "".join(value)
            if value:
                text_outputs.append(str(value))

    return "\n".join(text_outputs)


def first_number(text: str, label_pattern: str) -> float | None:
    match = re.search(
        rf"(?:{label_pattern})\s*:\s*({NUMBER})",
        text,
        flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def score_as_percentage(value: float) -> str:
    percentage = value * 100 if abs(value) <= 1 else value
    return f"{percentage:.1f}%"


def first_confusion_matrix(text: str) -> tuple[int, int, int, int] | None:
    match = re.search(
        r"\[\[\s*(\d+)[,\s]+(\d+)\s*\]\s*"
        r"\[\s*(\d+)[,\s]+(\d+)\s*\]\]",
        text,
    )
    return tuple(map(int, match.groups())) if match else None


def metrics_from_confusion_matrix(
    matrix: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    tn, fp, fn, tp = matrix
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    balanced_accuracy = (
        tn / (tn + fp) + tp / (tp + fn)
    ) / 2
    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return accuracy, balanced_accuracy, mcc


def repeated_metric(text: str, label: str, percentage: bool) -> str | None:
    match = re.search(
        rf"{re.escape(label)}\s*:\s*({NUMBER})\s*(?:\+/-|±)\s*({NUMBER})",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    mean, standard_deviation = map(float, match.groups())
    if percentage:
        return (
            f"{label}: {score_as_percentage(mean)} +/- "
            f"{score_as_percentage(standard_deviation)}"
        )
    return f"{label}: {mean:.3f} +/- {standard_deviation:.3f}"


def loco_summary(root: Path) -> list[str]:
    metrics_file = (
        root
        / "SupplementaryInformation"
        / "LOCO"
        / "validation_outputs"
        / "loco_metrics.csv"
    )
    if not metrics_file.is_file():
        return []

    lines = []
    with metrics_file.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                accuracy = score_as_percentage(float(row["Accuracy"]))
                balanced = score_as_percentage(float(row["Balanced accuracy"]))
                mcc = float(row["MCC"])
            except (KeyError, TypeError, ValueError):
                continue
            lines.append(
                f"{row.get('Protocol', 'LOCO')}: accuracy {accuracy}; "
                f"balanced accuracy {balanced}; MCC {mcc:.3f}"
            )
    return lines


def headline_results(
    workflow_name: str,
    executed_notebook: Path,
    root: Path,
) -> list[str]:
    """Extract a small, reader-facing result summary from each workflow."""
    if workflow_name == "generate_loco_splits":
        return []
    if workflow_name == "loco":
        return loco_summary(root)

    text = notebook_text_output(executed_notebook)
    if workflow_name == "shap":
        marker = "Metrics across the ten complete repeats:"
        section = text[text.rfind(marker):] if marker in text else text
        return [
            metric
            for metric in (
                repeated_metric(section, "Accuracy", percentage=True),
                repeated_metric(section, "Balanced accuracy", percentage=True),
                repeated_metric(section, "MCC", percentage=False),
            )
            if metric is not None
        ]

    if workflow_name == "test_train":
        accuracy = first_number(text, r"Validation accuracy")
        balanced = first_number(text, r"Balanced accuracy")
        mcc = first_number(text, r"Matthews correlation coefficient|MCC")
    elif workflow_name == "loo":
        accuracy = first_number(text, r"Accuracy")
        balanced = None
        mcc = None
    elif workflow_name == "pluskal":
        accuracy = first_number(text, r"Test accuracy")
        balanced = first_number(text, r"Balanced accuracy")
        mcc = first_number(text, r"Matthews correlation coefficient|MCC")
    else:
        return []

    matrix = first_confusion_matrix(text)
    if matrix is not None:
        matrix_accuracy, matrix_balanced, matrix_mcc = (
            metrics_from_confusion_matrix(matrix)
        )
        accuracy = accuracy if accuracy is not None else matrix_accuracy
        balanced = balanced if balanced is not None else matrix_balanced
        mcc = mcc if mcc is not None else matrix_mcc

    results = []
    if accuracy is not None:
        results.append(f"Accuracy: {score_as_percentage(accuracy)}")
    if balanced is not None:
        results.append(f"Balanced accuracy: {score_as_percentage(balanced)}")
    if mcc is not None:
        results.append(f"MCC: {mcc:.3f}")
    if matrix is not None:
        tn, fp, fn, tp = matrix
        results.append(
            "Confusion matrix [[TN, FP], [FN, TP]]: "
            f"[[{tn}, {fp}], [{fn}, {tp}]]"
        )
    return results


def print_headline_results(
    workflow_name: str,
    results: list[str],
    executed_notebook: Path,
    root: Path,
) -> None:
    title = RESULT_TITLES.get(workflow_name, workflow_name)
    print(f"\n{title} - headline results")
    if results:
        for result in results:
            print(f"  {result}")
    else:
        print("  See the executed notebook for the complete results.")
    print(f"  Full output: {executed_notebook.relative_to(root)}")


def save_results_summary(
    summaries: list[tuple[str, list[str], Path]],
    root: Path,
) -> Path:
    output = root / "reproduction_outputs" / "results_summary.txt"
    lines = ["Published model analyses - headline results", ""]

    for workflow_name, results, executed_notebook in summaries:
        title = RESULT_TITLES.get(workflow_name, workflow_name)
        lines.append(title)
        lines.extend(f"  {result}" for result in results)
        lines.append(f"  Full output: {executed_notebook.relative_to(root)}")
        lines.append("")

    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output


def main() -> None:
    arguments = parse_arguments()
    root = repository_root()
    selected = selected_workflows(root, arguments.analysis)

    if arguments.regenerate_loco_splits and "loco" not in selected:
        selected.append("loco")

    planned: list[tuple[str, Path]] = []
    if arguments.regenerate_loco_splits:
        generator = find_notebook(root, LOCO_GENERATORS)
        if generator is None:
            raise FileNotFoundError(
                "Generate_LOCO_Splits.ipynb was not found in the repository."
            )
        planned.append(("generate_loco_splits", generator))

    missing_notebooks: list[str] = []
    for workflow_name in selected:
        workflow = WORKFLOWS[workflow_name]
        notebook = find_notebook(root, workflow.notebooks)
        if notebook is None:
            missing_notebooks.append(
                f"- {workflow_name}: expected {' or '.join(workflow.notebooks)}"
            )
        else:
            planned.append((workflow_name, notebook))

    if missing_notebooks:
        raise FileNotFoundError(
            "Required analysis notebooks are missing:\n" + "\n".join(missing_notebooks)
        )

    check_datasets(root, selected)
    print("Published model analyses:")
    for workflow_name, notebook in planned:
        description = (
            "Regenerate published LOCO split files."
            if workflow_name == "generate_loco_splits"
            else WORKFLOWS[workflow_name].description
        )
        print(f"- {workflow_name}: {notebook.relative_to(root)}")
        print(f"  {description}")
    print("Bayesian feature and hyperparameter optimization is not run.")

    if arguments.dry_run:
        return

    check_dependencies(selected)
    output_directory = root / "reproduction_outputs" / "executed_notebooks"
    completed = []
    summaries = []
    try:
        for name, notebook in planned:
            executed = execute_notebook(
                notebook,
                name,
                root,
                output_directory,
                arguments.timeout,
            )
            completed.append(executed)
            results = headline_results(name, executed, root)
            summaries.append((name, results, executed))
            print_headline_results(name, results, executed, root)
    finally:
        for _, notebook in planned:
            pattern = f".{notebook.stem}.reproduction.*.ipynb"
            for temporary in notebook.parent.glob(pattern):
                temporary.unlink(missing_ok=True)

    print("\nExecuted notebooks:")
    for notebook in completed:
        print(f"- {notebook.relative_to(root)}")

    summary_file = save_results_summary(summaries, root)
    print(f"\nHeadline results saved to {summary_file.relative_to(root)}")


if __name__ == "__main__":
    main()
