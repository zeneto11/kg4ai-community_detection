from pathlib import Path


def get_run(base_output_path: Path) -> Path:
    """
    Determines the next run folder name like run1/, run2/, etc.
    """
    base_output_path.mkdir(parents=True, exist_ok=True)
    existing_runs = [
        int(p.name.replace("run", ""))
        for p in base_output_path.iterdir()
        if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit()
    ]
    next_run_number = max(existing_runs, default=0) + 1
    next_run_path = base_output_path / f"run{next_run_number}"
    next_run_path.mkdir()
    return next_run_path
