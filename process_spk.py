import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import shutil

    import marimo as mo

    return mo, shutil


@app.cell
def _(mo, shutil):
    octave_path = shutil.which("octave")

    status = (
        f"Octave available on PATH: `{octave_path}`"
        if octave_path
        else "Octave command not found on PATH."
    )

    mo.md(status)
    return


if __name__ == "__main__":
    app.run()
