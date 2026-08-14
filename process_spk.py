import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path
    import shutil
    from string import Template
    import subprocess

    import marimo as mo

    def quote_for_eval(value: Path | str) -> str:
        return str(value).replace("'", "''")

    def build_runtime_log_path(spk_path: Path, runtime: str) -> Path:
        return spk_path.with_name(f"{spk_path.stem}_{runtime}.log")

    def read_log_tail(log_path: Path, *, max_lines: int = 200) -> str:
        if not log_path.is_file():
            return ""

        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""

        return "\n".join(lines[-max_lines:])

    def octave_loader_is_compatible(loader_dir: Path) -> bool:
        spike_dataset_file = loader_dir / "SpikeDataSet.m"
        heterogeneous_shim = loader_dir / "+matlab" / "+mixin" / "Heterogeneous.m"
        custom_display_shim = loader_dir / "+matlab" / "+mixin" / "CustomDisplay.m"

        if not spike_dataset_file.is_file():
            return False

        try:
            spike_dataset_text = spike_dataset_file.read_text(
                encoding="utf-8", errors="ignore"
            )
        except OSError:
            return False

        return (
            "LoadAllSpikesDetailed" in spike_dataset_text
            and heterogeneous_shim.is_file()
            and custom_display_shim.is_file()
        )

    def get_runtime_messages(
        *,
        runtime: str,
        matlab_bin: str | None,
        octave_bin: str | None,
        octave_loader_compatible: bool,
    ) -> tuple[list[str], list[str]]:
        notes: list[str] = []
        blockers: list[str] = []

        if runtime == "matlab":
            if not matlab_bin:
                blockers.append(
                    "MATLAB is selected but no usable MATLAB executable is configured."
                )
            return notes, blockers

        if runtime == "octave":
            notes.append(
                "Octave support is experimental and developer-only. "
                "Switch `mea-rasterplotter/vendor/AxionFileLoader` to the "
                "`feature/octave` branch before running it."
            )
            if not octave_bin:
                blockers.append("Octave is selected but `octave` was not found on PATH.")
            if not octave_loader_compatible:
                blockers.append(
                    "The current vendored AxionFileLoader does not appear to be "
                    "the Octave-compatible branch."
                )
            return notes, blockers

        blockers.append(f"Unsupported runtime selection: {runtime}")
        return notes, blockers

    def parse_env_overrides(raw_text: str) -> tuple[dict[str, str], list[str]]:
        overrides: dict[str, str] = {}
        errors: list[str] = []

        for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            key_part, separator, value_part = raw_line.partition("=")
            if separator != "=":
                errors.append(
                    f"MATLAB environment line {line_number} must use `KEY=VALUE` format."
                )
                continue

            key = key_part.strip()
            if not key:
                errors.append(
                    f"MATLAB environment line {line_number} is missing a variable name."
                )
                continue

            if not (key[0].isalpha() or key[0] == "_") or any(
                not (char.isalnum() or char == "_") for char in key[1:]
            ):
                errors.append(
                    f"MATLAB environment variable `{key}` on line {line_number} is invalid."
                )
                continue

            overrides[key] = value_part.lstrip()

        return overrides, errors

    def resolve_matlab_executable(
        *,
        configured_path: str,
        detected_path: str | None,
        env_overrides: dict[str, str],
    ) -> tuple[str | None, list[str], bool]:
        merged_env = os.environ.copy()
        merged_env.update(env_overrides)

        configured_path = configured_path.strip()
        if not configured_path:
            discovered_path = shutil.which("matlab", path=merged_env.get("PATH"))
            return discovered_path or detected_path, [], False

        expanded_path = Template(configured_path).safe_substitute(merged_env)
        candidate = Path(expanded_path).expanduser()

        if not candidate.is_absolute():
            return (
                None,
                [
                    "Configured MATLAB executable path must be absolute. "
                    "You can use environment variables inside it, for example "
                    "`$MATLAB_HOME/bin/matlab`."
                ],
                True,
            )

        resolved_candidate = candidate.resolve()
        if not resolved_candidate.exists():
            return (
                None,
                [f"Configured MATLAB executable was not found: `{resolved_candidate}`"],
                True,
            )
        if resolved_candidate.is_dir():
            return (
                None,
                [f"Configured MATLAB executable points to a directory: `{resolved_candidate}`"],
                True,
            )
        if not os.access(resolved_candidate, os.X_OK):
            return (
                None,
                [f"Configured MATLAB executable is not executable: `{resolved_candidate}`"],
                True,
            )

        return str(resolved_candidate), [], True

    def run_axisfile_wrapper_with_matlab(
        *,
        spk_path: Path,
        wrapper_script: Path,
        loader_dir: Path,
        matlab_bin: str,
        matlab_env_overrides: dict[str, str],
    ) -> tuple[Path, str, str, Path]:
        spk_path = spk_path.expanduser().resolve()
        wrapper_script = wrapper_script.expanduser().resolve()
        loader_dir = loader_dir.expanduser().resolve()
        csv_path = spk_path.with_suffix(".csv")
        log_path = build_runtime_log_path(spk_path, "matlab")

        if not spk_path.is_file():
            raise FileNotFoundError(f"SPK file not found: {spk_path}")
        if not wrapper_script.is_file():
            raise FileNotFoundError(f"Wrapper script not found: {wrapper_script}")
        if not loader_dir.is_dir():
            raise FileNotFoundError(f"AxionFileLoader directory not found: {loader_dir}")
        if not matlab_bin:
            raise RuntimeError("MATLAB was not found on PATH.")

        batch_code = (
            f"addpath('{quote_for_eval(wrapper_script.parent)}');"
            f"spk_path='{quote_for_eval(spk_path)}';"
            f"output_csv='{quote_for_eval(csv_path)}';"
            f"loader_dir='{quote_for_eval(loader_dir)}';"
            "try;"
            "extract_spk_with_axisfile_matlab(spk_path, output_csv, loader_dir);"
            "catch ME;"
            "fprintf(2, '%s\\n', getReport(ME, 'extended', 'hyperlinks', 'off'));"
            "exit(1);"
            "end;"
        )

        subprocess_env = os.environ.copy()
        subprocess_env.update(matlab_env_overrides)

        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            result = subprocess.run(
                [matlab_bin, "-batch", batch_code],
                env=subprocess_env,
                stderr=subprocess.STDOUT,
                stdout=log_file,
                text=True,
                check=False,
            )
        log_tail = read_log_tail(log_path)

        if result.returncode != 0:
            raise RuntimeError(
                "MATLAB conversion failed.\n"
                f"Log file: {log_path}\n"
                f"LOG TAIL:\n{log_tail}"
            )

        if not csv_path.exists():
            raise RuntimeError(
                "MATLAB conversion finished but CSV was not created.\n"
                f"CSV path: {csv_path}\n"
                f"Log file: {log_path}\n"
                f"LOG TAIL:\n{log_tail}"
            )

        return csv_path, log_tail, "", log_path

    def run_axisfile_wrapper_with_octave(
        *,
        spk_path: Path,
        wrapper_script: Path,
        loader_dir: Path,
        octave_bin: str,
    ) -> tuple[Path, str, str, Path]:
        spk_path = spk_path.expanduser().resolve()
        wrapper_script = wrapper_script.expanduser().resolve()
        loader_dir = loader_dir.expanduser().resolve()
        csv_path = spk_path.with_suffix(".csv")
        log_path = build_runtime_log_path(spk_path, "octave")

        if not spk_path.is_file():
            raise FileNotFoundError(f"SPK file not found: {spk_path}")
        if not wrapper_script.is_file():
            raise FileNotFoundError(f"Wrapper script not found: {wrapper_script}")
        if not loader_dir.is_dir():
            raise FileNotFoundError(f"AxionFileLoader directory not found: {loader_dir}")
        if not octave_bin:
            raise RuntimeError("Octave was not found on PATH.")

        eval_code = (
            f"addpath('{quote_for_eval(wrapper_script.parent)}');"
            f"spk_path='{quote_for_eval(spk_path)}';"
            f"output_csv='{quote_for_eval(csv_path)}';"
            f"loader_dir='{quote_for_eval(loader_dir)}';"
            "try;"
            "extract_spk_with_axisfile_octave(spk_path, output_csv, loader_dir);"
            "catch ME;"
            "fprintf(2, 'ERROR: %s\\n', ME.message);"
            "for k = 1:numel(ME.stack);"
            "fprintf(2, '  at %s:%d\\n', ME.stack(k).file, ME.stack(k).line);"
            "end;"
            "exit(1);"
            "end;"
        )

        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            result = subprocess.run(
                [octave_bin, "--quiet", "--no-gui", "--no-history", "--eval", eval_code],
                stderr=subprocess.STDOUT,
                stdout=log_file,
                text=True,
                check=False,
            )
        log_tail = read_log_tail(log_path)

        if result.returncode != 0:
            hint = ""
            error_text = log_tail
            if "matlab.mixin" in error_text or "no such method or property 'empty'" in error_text:
                hint = (
                    "\nHint: AxionFileLoader relies on MATLAB class features not fully supported "
                    "by this Octave build."
                )
            raise RuntimeError(
                "Octave conversion failed.\n"
                f"Log file: {log_path}\n"
                f"LOG TAIL:\n{log_tail}"
                f"{hint}"
            )

        if not csv_path.exists():
            raise RuntimeError(
                "Octave conversion finished but CSV was not created.\n"
                f"CSV path: {csv_path}\n"
                f"Log file: {log_path}\n"
                f"LOG TAIL:\n{log_tail}"
            )

        return csv_path, log_tail, "", log_path

    return (
        Path,
        build_runtime_log_path,
        get_runtime_messages,
        mo,
        octave_loader_is_compatible,
        parse_env_overrides,
        read_log_tail,
        resolve_matlab_executable,
        run_axisfile_wrapper_with_matlab,
        run_axisfile_wrapper_with_octave,
        shutil,
    )


@app.cell
def _(Path):
    notebook_dir = Path(__file__).resolve().parent

    matlab_wrapper_script = notebook_dir / "extract_spk_with_axisfile_matlab.m"
    octave_wrapper_script = notebook_dir / "extract_spk_with_axisfile_octave.m"
    loader_dir = notebook_dir / "vendor" / "AxionFileLoader" / "AxionFileLoader"
    initial_spk_dir = notebook_dir
    return (
        initial_spk_dir,
        loader_dir,
        matlab_wrapper_script,
        octave_wrapper_script,
    )


@app.cell
def _(mo):
    show_spk_picker, set_show_spk_picker = mo.state(True)
    return set_show_spk_picker, show_spk_picker


@app.cell
def _(mo):
    selected_runtime, set_selected_runtime = mo.state("matlab")
    return selected_runtime, set_selected_runtime


@app.cell
def _(matlab_bin_on_path, mo):
    matlab_executable_path_input = mo.ui.text(
        label="MATLAB executable path override",
        value="",
        placeholder=matlab_bin_on_path or "/absolute/path/to/matlab",
        full_width=True,
    )
    matlab_env_overrides_input = mo.ui.text_area(
        label="MATLAB environment overrides",
        value="",
        placeholder="# One KEY=VALUE per line\nLD_LIBRARY_PATH=/opt/matlab/runtime\nMATLAB_HOME=/opt/MATLAB/R2024b",
        full_width=True,
    )
    return matlab_env_overrides_input, matlab_executable_path_input


@app.cell
def _(shutil):
    matlab_bin_on_path = shutil.which("matlab")
    octave_bin_on_path = shutil.which("octave")
    return matlab_bin_on_path, octave_bin_on_path


@app.cell
def _(selected_runtime):
    selected_runtime_value = selected_runtime()
    return (selected_runtime_value,)


@app.cell
def _(
    initial_spk_dir,
    mo,
    selected_runtime_value,
    set_selected_runtime,
    set_show_spk_picker,
):
    matlab_runtime_button = mo.ui.button(
        label="MATLAB selected" if selected_runtime_value == "matlab" else "Use MATLAB",
        on_click=lambda _: set_selected_runtime(lambda _: "matlab"),
    )
    octave_runtime_button = mo.ui.button(
        label="Octave selected" if selected_runtime_value == "octave" else "Use Octave",
        on_click=lambda _: set_selected_runtime(lambda _: "octave"),
    )
    spk_path_toggle = mo.ui.button(
        label="Choose/change SPK file",
        on_click=lambda _: set_show_spk_picker(lambda current: not current),
    )
    spk_path_picker = mo.ui.file_browser(
        initial_path=initial_spk_dir,
        filetypes=[".spk"],
        multiple=False,
        label="SPK file",
        on_change=lambda _: set_show_spk_picker(False),
    )
    return (
        matlab_runtime_button,
        octave_runtime_button,
        spk_path_picker,
        spk_path_toggle,
    )


@app.cell
def _(Path, spk_path_picker):
    selected_spk_path = spk_path_picker.path(0)
    spk_path = Path(selected_spk_path) if selected_spk_path else None
    output_csv_path = spk_path.with_suffix(".csv") if spk_path else None
    return output_csv_path, spk_path


@app.cell
def _(
    build_runtime_log_path,
    get_runtime_messages,
    loader_dir,
    matlab_bin_on_path,
    matlab_env_overrides_input,
    matlab_executable_path_input,
    matlab_runtime_button,
    matlab_wrapper_script,
    mo,
    octave_bin_on_path,
    octave_loader_is_compatible,
    octave_runtime_button,
    octave_wrapper_script,
    output_csv_path,
    parse_env_overrides,
    resolve_matlab_executable,
    selected_runtime_value,
    show_spk_picker,
    spk_path,
    spk_path_picker,
    spk_path_toggle,
):
    matlab_env_overrides, matlab_env_errors = parse_env_overrides(
        matlab_env_overrides_input.value
    )
    matlab_bin, matlab_path_errors, has_matlab_override = resolve_matlab_executable(
        configured_path=matlab_executable_path_input.value,
        detected_path=matlab_bin_on_path,
        env_overrides=matlab_env_overrides,
    )
    octave_bin = octave_bin_on_path
    octave_loader_compatible = octave_loader_is_compatible(loader_dir)
    runtime_notes, runtime_blockers = get_runtime_messages(
        runtime=selected_runtime_value,
        matlab_bin=matlab_bin,
        octave_bin=octave_bin,
        octave_loader_compatible=octave_loader_compatible,
    )
    config_warnings: list[str] = []
    for warning in matlab_path_errors + matlab_env_errors + runtime_blockers:
        if warning not in config_warnings:
            config_warnings.append(warning)
    if selected_runtime_value == "matlab":
        runtime_blockers = config_warnings

    active_wrapper_script = (
        matlab_wrapper_script
        if selected_runtime_value == "matlab"
        else octave_wrapper_script
    )
    active_log_path = (
        build_runtime_log_path(spk_path, selected_runtime_value)
        if spk_path is not None
        else None
    )
    can_extract = (
        spk_path is not None
        and spk_path.is_file()
        and spk_path.suffix.lower() == ".spk"
        and not runtime_blockers
    )
    extract_spikes_button = mo.ui.run_button(
        label="Extract spike timings to CSV",
        disabled=not can_extract,
        kind="success",
    )

    status = [
        f"- Selected runtime: `{selected_runtime_value}`",
        (
            f"- MATLAB binary on PATH: `{matlab_bin_on_path}`"
            if matlab_bin_on_path
            else "- MATLAB binary on PATH: not found"
        ),
        (
            f"- MATLAB executable override: `{matlab_executable_path_input.value}`"
            if matlab_executable_path_input.value.strip()
            else "- MATLAB executable override: none"
        ),
        (
            f"- Effective MATLAB executable: `{matlab_bin}`"
            if matlab_bin
            else "- Effective MATLAB executable: not configured"
        ),
        (
            "- MATLAB executable source: override"
            if has_matlab_override
            else "- MATLAB executable source: PATH lookup"
        ),
        (
            f"- MATLAB environment overrides: {len(matlab_env_overrides)} configured"
            if matlab_env_overrides
            else "- MATLAB environment overrides: none"
        ),
        f"- Octave binary: `{octave_bin}`" if octave_bin else "- Octave binary: not found on PATH",
        f"- MATLAB wrapper: `{matlab_wrapper_script}`",
        f"- Octave wrapper: `{octave_wrapper_script}`",
        f"- Active wrapper: `{active_wrapper_script}`",
        (
            f"- Active log file: `{active_log_path}`"
            if active_log_path is not None
            else "- Active log file: not available until an SPK file is selected"
        ),
        f"- Loader directory: `{loader_dir}`",
        (
            "- Octave loader compatibility: detected"
            if octave_loader_compatible
            else "- Octave loader compatibility: not detected in current vendor checkout"
        ),
    ]

    blocks = [mo.md("## Configuration\n" + "\n".join(status))]
    if runtime_notes:
        blocks.append(mo.md("**Notes**\n" + "\n".join(f"- {note}" for note in runtime_notes)))
    blocks.append(
        mo.vstack(
            [
                matlab_executable_path_input,
                matlab_env_overrides_input,
            ]
        )
    )
    if config_warnings:
        blocks.append(
            mo.md("**Warnings**\n" + "\n".join(f"- {warning}" for warning in config_warnings))
        )

    blocks.append(mo.hstack([matlab_runtime_button, octave_runtime_button], align="start"))
    file_controls = [spk_path_toggle]
    if show_spk_picker():
        file_controls.append(spk_path_picker)
    blocks.append(mo.vstack(file_controls, align="stretch", gap=0.3))

    if spk_path is None:
        displayed_paths = (
            "**Input .spk file:** None selected\n\n"
            "**Output .csv file:** Not available"
        )
    else:
        displayed_paths = (
            f"**Input .spk file:** `{spk_path}`\n\n"
            f"**Output .csv file:** `{output_csv_path}`"
        )
    blocks.append(mo.md(displayed_paths))
    blocks.append(extract_spikes_button)

    mo.vstack(blocks, align="stretch", gap=0.5)
    return (
        active_log_path,
        active_wrapper_script,
        extract_spikes_button,
        matlab_bin,
        matlab_env_overrides,
        octave_bin,
        runtime_blockers,
    )


@app.cell
def _(
    active_log_path,
    active_wrapper_script,
    extract_spikes_button,
    loader_dir,
    matlab_bin,
    matlab_env_overrides,
    mo,
    octave_bin,
    read_log_tail,
    run_axisfile_wrapper_with_matlab,
    run_axisfile_wrapper_with_octave,
    runtime_blockers,
    selected_runtime_value,
    spk_path,
):
    mo.stop(not extract_spikes_button.value)

    if spk_path is None or active_log_path is None:
        message = (
            "## Conversion Result\n"
            "- Select a valid .spk file before starting extraction."
        )
    elif runtime_blockers:
        message = (
            "## Conversion Result\n"
            + "\n".join(f"- {warning}" for warning in runtime_blockers)
            + f"\n- Expected log file: `{active_log_path}`\n"
        )
    else:
        with mo.status.spinner(
            title="Extracting spike timings to CSV...",
            subtitle=f"{spk_path.name} using {selected_runtime_value}",
        ):
            try:
                if selected_runtime_value == "matlab":
                    csv_path, stdout, stderr, log_path = run_axisfile_wrapper_with_matlab(
                        spk_path=spk_path,
                        wrapper_script=active_wrapper_script,
                        loader_dir=loader_dir,
                        matlab_bin=matlab_bin,
                        matlab_env_overrides=matlab_env_overrides,
                    )
                else:
                    csv_path, stdout, stderr, log_path = run_axisfile_wrapper_with_octave(
                        spk_path=spk_path,
                        wrapper_script=active_wrapper_script,
                        loader_dir=loader_dir,
                        octave_bin=octave_bin,
                    )

                message = (
                    "## Conversion Result\n"
                    f"- Runtime: `{selected_runtime_value}`\n"
                    f"- CSV created at: `{csv_path}`\n"
                    f"- Log file: `{log_path}`\n"
                )
                if stdout:
                    message += f"```text\n{stdout}\n```\n"
                if stderr:
                    message += f"```text\n{stderr}\n```\n"
            except Exception as exc:
                log_tail = read_log_tail(active_log_path)
                message = (
                    "## Conversion Result\n"
                    f"- Runtime: `{selected_runtime_value}`\n"
                    "- Status: failed\n"
                    f"- Log file: `{active_log_path}`\n"
                    f"```text\n{exc}\n```\n"
                )
                if log_tail:
                    message += f"### Log Tail\n```text\n{log_tail}\n```\n"

    mo.md(message)
    return


if __name__ == "__main__":
    app.run()
