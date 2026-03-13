import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import shutil
    import subprocess

    import marimo as mo

    def quote_for_octave(value: Path | str) -> str:
        return str(value).replace("'", "''")

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
            f"addpath('{quote_for_octave(wrapper_script.parent)}');"
            f"spk_path='{quote_for_octave(spk_path)}';"
            f"output_csv='{quote_for_octave(csv_path)}';"
            f"loader_dir='{quote_for_octave(loader_dir)}';"
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

    return Path, mo, run_axisfile_wrapper_with_octave, shutil


@app.cell
def _(Path):
    notebook_dir = Path(__file__).resolve().parent

    wrapper_script = notebook_dir / "extract_spk_with_axisfile_octave.m"
    loader_dir = notebook_dir / "vendor" / "AxionFileLoader" / "AxionFileLoader"
    default_spk_path = (
        notebook_dir.parent
        / "data"
        / "201023_LvM_256086_1268-20_MEA_rCortex_Permethrin_baseline_female_DIV11(000)_Spike Detector (7 x STD)(000).spk"
    )
    return default_spk_path, loader_dir, wrapper_script


@app.cell
def _(default_spk_path, mo):
    show_spk_picker, set_show_spk_picker = mo.state(not default_spk_path.is_file())
    return set_show_spk_picker, show_spk_picker


@app.cell
def _(default_spk_path, mo, set_show_spk_picker):
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
    return spk_path_picker, spk_path_toggle


@app.cell
def _(default_spk_path, spk_path_picker):
    spk_path = spk_path_picker.path(0) or default_spk_path
    output_csv_path = spk_path.with_suffix(".csv")
    return output_csv_path, spk_path


@app.cell
def _(
    loader_dir,
    mo,
    output_csv_path,
    show_spk_picker,
    shutil,
    spk_path,
    spk_path_picker,
    spk_path_toggle,
    wrapper_script,
):
    octave_bin = shutil.which("octave")

    status = [
        f"- Octave binary: `{octave_bin}`" if octave_bin else "- Octave binary: not found on PATH",
        f"- Wrapper script: `{wrapper_script}`",
        f"- Loader directory: `{loader_dir}`",
        f"- Input .spk file: `{spk_path}`",
        f"- Output .csv file: `{output_csv_path}`",
    ]

    controls = [spk_path_toggle]
    if show_spk_picker():
        controls.append(spk_path_picker)

    mo.vstack(
        [
            mo.md("## Configuration\n" + "\n".join(status)),
            mo.hstack(controls, align="start"),
        ]
    )
    return (octave_bin,)


@app.cell
def _(
    loader_dir,
    mo,
    octave_bin,
    run_axisfile_wrapper_with_octave,
    spk_path,
    wrapper_script,
):
    csv_path, stdout, stderr = run_axisfile_wrapper_with_octave(
        spk_path=spk_path,
        wrapper_script=wrapper_script,
        loader_dir=loader_dir,
        octave_bin=octave_bin,
    )

    message = (
        "## Conversion Result\n"
        f"- CSV created at: `{csv_path}`\n"
    )

    if stdout:
        message += f"```text\n{stdout}\n```\n"
    if stderr:
        message += f"```text\n{stderr}\n```\n"

    mo.md(message)
    return


if __name__ == "__main__":
    app.run()
