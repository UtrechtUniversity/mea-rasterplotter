import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import shutil
    import subprocess

    import marimo as mo

    def quote_for_eval(value: Path | str) -> str:
        return str(value).replace("'", "''")

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
                blockers.append("MATLAB is selected but `matlab` was not found on PATH.")
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

    def run_axisfile_wrapper_with_matlab(
        *,
        spk_path: Path,
        wrapper_script: Path,
        loader_dir: Path,
        matlab_bin: str,
    ) -> tuple[Path, str, str]:
        spk_path = spk_path.expanduser().resolve()
        wrapper_script = wrapper_script.expanduser().resolve()
        loader_dir = loader_dir.expanduser().resolve()
        csv_path = spk_path.with_suffix(".csv")

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

        result = subprocess.run(
            [matlab_bin, "-batch", batch_code],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "MATLAB conversion failed.\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        if not csv_path.exists():
            raise RuntimeError(f"Conversion finished but CSV was not created: {csv_path}")

        return csv_path, result.stdout.strip(), result.stderr.strip()

    def run_axisfile_wrapper_with_octave(
        *,
        spk_path: Path,
        wrapper_script: Path,
        loader_dir: Path,
        octave_bin: str,
    ) -> tuple[Path, str, str]:
        spk_path = spk_path.expanduser().resolve()
        wrapper_script = wrapper_script.expanduser().resolve()
        loader_dir = loader_dir.expanduser().resolve()
        csv_path = spk_path.with_suffix(".csv")

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

        result = subprocess.run(
            [octave_bin, "--quiet", "--no-gui", "--no-history", "--eval", eval_code],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            hint = ""
            error_text = f"{result.stdout}\n{result.stderr}"
            if "matlab.mixin" in error_text or "no such method or property 'empty'" in error_text:
                hint = (
                    "\nHint: AxionFileLoader relies on MATLAB class features not fully supported "
                    "by this Octave build."
                )
            raise RuntimeError(
                "Octave conversion failed.\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
                f"{hint}"
            )

        if not csv_path.exists():
            raise RuntimeError(f"Conversion finished but CSV was not created: {csv_path}")

        return csv_path, result.stdout.strip(), result.stderr.strip()

    return (
        Path,
        get_runtime_messages,
        mo,
        octave_loader_is_compatible,
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
    default_spk_path = (
        notebook_dir.parent
        / "data"
        / "201023_LvM_256086_1268-20_MEA_rCortex_Permethrin_baseline_female_DIV11(000)_Spike Detector (7 x STD)(000).spk"
    )
    return (
        default_spk_path,
        loader_dir,
        matlab_wrapper_script,
        octave_wrapper_script,
    )


@app.cell
def _(default_spk_path, mo):
    show_spk_picker, set_show_spk_picker = mo.state(not default_spk_path.is_file())
    return set_show_spk_picker, show_spk_picker


@app.cell
def _(mo):
    selected_runtime, set_selected_runtime = mo.state("matlab")
    return selected_runtime, set_selected_runtime


@app.cell
def _(selected_runtime):
    selected_runtime_value = selected_runtime()
    return (selected_runtime_value,)


@app.cell
def _(
    default_spk_path,
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
        initial_path=default_spk_path.parent,
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
def _(default_spk_path, spk_path_picker):
    spk_path = spk_path_picker.path(0) or default_spk_path
    output_csv_path = spk_path.with_suffix(".csv")
    return output_csv_path, spk_path


@app.cell
def _(
    get_runtime_messages,
    loader_dir,
    matlab_runtime_button,
    matlab_wrapper_script,
    mo,
    octave_loader_is_compatible,
    octave_runtime_button,
    octave_wrapper_script,
    output_csv_path,
    selected_runtime_value,
    show_spk_picker,
    shutil,
    spk_path,
    spk_path_picker,
    spk_path_toggle,
):
    matlab_bin = shutil.which("matlab")
    octave_bin = shutil.which("octave")
    octave_loader_compatible = octave_loader_is_compatible(loader_dir)
    runtime_notes, runtime_blockers = get_runtime_messages(
        runtime=selected_runtime_value,
        matlab_bin=matlab_bin,
        octave_bin=octave_bin,
        octave_loader_compatible=octave_loader_compatible,
    )
    active_wrapper_script = (
        matlab_wrapper_script
        if selected_runtime_value == "matlab"
        else octave_wrapper_script
    )

    status = [
        f"- Selected runtime: `{selected_runtime_value}`",
        f"- MATLAB binary: `{matlab_bin}`" if matlab_bin else "- MATLAB binary: not found on PATH",
        f"- Octave binary: `{octave_bin}`" if octave_bin else "- Octave binary: not found on PATH",
        f"- MATLAB wrapper: `{matlab_wrapper_script}`",
        f"- Octave wrapper: `{octave_wrapper_script}`",
        f"- Active wrapper: `{active_wrapper_script}`",
        f"- Loader directory: `{loader_dir}`",
        (
            "- Octave loader compatibility: detected"
            if octave_loader_compatible
            else "- Octave loader compatibility: not detected in current vendor checkout"
        ),
        f"- Input .spk file: `{spk_path}`",
        f"- Output .csv file: `{output_csv_path}`",
    ]

    blocks = [mo.md("## Configuration\n" + "\n".join(status))]
    if runtime_notes:
        blocks.append(mo.md("**Notes**\n" + "\n".join(f"- {note}" for note in runtime_notes)))
    if runtime_blockers:
        blocks.append(
            mo.md("**Warnings**\n" + "\n".join(f"- {warning}" for warning in runtime_blockers))
        )

    controls = [matlab_runtime_button, octave_runtime_button, spk_path_toggle]
    if show_spk_picker():
        controls.append(spk_path_picker)

    blocks.append(mo.hstack(controls, align="start"))
    mo.vstack(blocks)
    return active_wrapper_script, matlab_bin, octave_bin, runtime_blockers


@app.cell
def _(
    active_wrapper_script,
    loader_dir,
    matlab_bin,
    mo,
    octave_bin,
    run_axisfile_wrapper_with_matlab,
    run_axisfile_wrapper_with_octave,
    runtime_blockers,
    selected_runtime_value,
    spk_path,
):
    if runtime_blockers:
        message = (
            "## Conversion Result\n"
            + "\n".join(f"- {warning}" for warning in runtime_blockers)
        )
    else:
        if selected_runtime_value == "matlab":
            csv_path, stdout, stderr = run_axisfile_wrapper_with_matlab(
                spk_path=spk_path,
                wrapper_script=active_wrapper_script,
                loader_dir=loader_dir,
                matlab_bin=matlab_bin,
            )
        else:
            csv_path, stdout, stderr = run_axisfile_wrapper_with_octave(
                spk_path=spk_path,
                wrapper_script=active_wrapper_script,
                loader_dir=loader_dir,
                octave_bin=octave_bin,
            )

        message = (
            "## Conversion Result\n"
            f"- Runtime: `{selected_runtime_value}`\n"
            f"- CSV created at: `{csv_path}`\n"
        )

        if stdout:
            message += f"```text\n{stdout}\n```\n"
        if stderr:
            message += f"```text\n{stderr}\n```\n"

    mo.md(message)


if __name__ == "__main__":
    app.run()
