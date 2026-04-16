import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell
def _():
    from dataclasses import dataclass
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    SPIKE_REQUIRED_COLUMNS = {
        "Channel_Label",
        "Well_Label",
        "Timestamp",
        "Maximum_Amplitude",
        "Minimum_Amplitude",
        "Peak_to_peak_Amplitude",
    }
    @dataclass
    class PlotSettings:
        start_time: float
        end_time: float
        figure_width: float
        figure_height: float
        line_length: float
        line_width: float
        color: str
        x_pad_left: float
        x_pad_right: float
        show_channel_labels: bool

    def assert_required_columns(
        df: pl.DataFrame, required: set[str], dataset_name: str
    ) -> None:
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(
                f"{dataset_name} is missing required columns: {', '.join(missing)}"
            )

    def load_spike_csv(path: Path) -> pl.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(f"Spike CSV not found: {path}")
        spikes = pl.read_csv(path)
        assert_required_columns(spikes, SPIKE_REQUIRED_COLUMNS, "Spike CSV")
        return spikes.with_columns(
            pl.col("Well_Label").cast(pl.Utf8, strict=False).alias("Well_Label"),
            pl.col("Timestamp").cast(pl.Float64, strict=False).alias("Timestamp"),
            pl.col("Channel_Label").cast(pl.Utf8).alias("Channel_Label"),
        )

    def available_wells(df: pl.DataFrame) -> list[str]:
        return (
            df.select(pl.col("Well_Label").drop_nulls().unique(maintain_order=True))
            .to_series()
            .to_list()
        )

    def filter_well_window(
        df: pl.DataFrame, well_label: str, start_time: float, end_time: float
    ) -> pl.DataFrame:
        window_start = min(start_time, end_time)
        window_end = max(start_time, end_time)
        return df.filter(
            (pl.col("Well_Label") == well_label)
            & pl.col("Timestamp").is_not_null()
            & pl.col("Channel_Label").is_not_null()
            & (pl.col("Timestamp") >= window_start)
            & (pl.col("Timestamp") <= window_end)
        )

    def build_event_series(df: pl.DataFrame) -> tuple[list[np.ndarray], list[str]]:
        channel_order = (
            df.select(pl.col("Channel_Label").unique(maintain_order=True))
            .to_series()
            .to_list()
        )
        events: list[np.ndarray] = []
        channel_labels: list[str] = []

        for channel in channel_order:
            timestamps = (
                df.filter(pl.col("Channel_Label") == channel)
                .select("Timestamp")
                .to_series()
                .drop_nulls()
                .to_numpy()
            )
            if timestamps.size == 0:
                continue
            events.append(np.asarray(timestamps, dtype=float))
            channel_labels.append(str(channel))

        return events, channel_labels

    def render_raster(
        events: list[np.ndarray],
        channel_labels: list[str],
        well_label: str,
        settings: PlotSettings,
    ):
        window_start = min(settings.start_time, settings.end_time)
        window_end = max(settings.start_time, settings.end_time)
        fig, ax = plt.subplots(
            figsize=(settings.figure_width, settings.figure_height),
            constrained_layout=True,
        )

        if events:
            line_offsets = np.arange(1, len(events) + 1, dtype=float).tolist()
            ax.eventplot(
                events,
                orientation="horizontal",
                lineoffsets=line_offsets,
                linelengths=settings.line_length,
                linewidths=settings.line_width,
                colors=settings.color,
            )
            ax.set_ylim(0.5, len(events) + 0.5)
            if settings.show_channel_labels:
                ax.set_yticks(line_offsets)
                ax.set_yticklabels(channel_labels, fontsize=9)
            else:
                ax.set_yticks([])
        else:
            ax.set_yticks([])
            ax.text(
                0.5,
                0.5,
                "No spikes in selected well/time window.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        ax.set_xlim(window_start - settings.x_pad_left, window_end + settings.x_pad_right)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("")
        ax.set_title(f"Raster Plot for Well {well_label}")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        return fig

    return (
        Path,
        PlotSettings,
        available_wells,
        build_event_series,
        filter_well_window,
        load_spike_csv,
        mo,
        np,
        pl,
        render_raster,
    )


@app.cell
def _(Path):
    notebook_dir = Path(__file__).resolve().parent
    default_spike_csv = (
        notebook_dir
    )
    return (default_spike_csv,)


@app.cell
def _(default_spike_csv, mo):
    show_spike_picker, set_show_spike_picker = mo.state(
        not default_spike_csv.is_file()
    )
    return set_show_spike_picker, show_spike_picker


@app.cell
def _(default_spike_csv, mo, set_show_spike_picker):
    spike_csv_toggle = mo.ui.button(
        label="Choose/change spike CSV",
        on_click=lambda _: set_show_spike_picker(lambda current: not current),
    )
    spike_csv_path = mo.ui.file_browser(
        initial_path=default_spike_csv.parent,
        filetypes=[".csv"],
        multiple=False,
        label="Spike CSV file",
        on_change=lambda _: set_show_spike_picker(False),
    )
    return spike_csv_path, spike_csv_toggle


@app.cell
def _(
    default_spike_csv,
    mo,
    show_spike_picker,
    spike_csv_path,
    spike_csv_toggle,
):
    displayed_spike_csv = spike_csv_path.path(0) or (
        default_spike_csv if default_spike_csv.is_file() else ""
    )
    input_widgets = [
        mo.md("### Inputs"),
        mo.md(f"Selected spike CSV: `{displayed_spike_csv}`"),
        spike_csv_toggle,
    ]
    if show_spike_picker():
        input_widgets.append(spike_csv_path)
    if not displayed_spike_csv:
        input_widgets.append(
            mo.md("Select a spike CSV to continue.").callout(kind="warn")
        )
    mo.vstack(input_widgets, align="stretch", gap=0.3)
    return


@app.cell
def _(Path, default_spike_csv, mo, spike_csv_path):
    selected_spike_csv_path = spike_csv_path.path(0)
    resolved_spike_csv = (
        Path(selected_spike_csv_path)
        if selected_spike_csv_path
        else (default_spike_csv if default_spike_csv.is_file() else None)
    )
    mo.stop(resolved_spike_csv is None)
    return (resolved_spike_csv,)


@app.cell
def _(load_spike_csv, mo, resolved_spike_csv):
    mo.stop(resolved_spike_csv is None)
    rasterplot_data = load_spike_csv(resolved_spike_csv)
    spike_csv = resolved_spike_csv
    return rasterplot_data, spike_csv


@app.cell
def _(available_wells, rasterplot_data):
    wells = available_wells(rasterplot_data)
    return (wells,)


@app.cell
def _(np, pl, rasterplot_data):
    timestamp_bounds = rasterplot_data.select(
        pl.col("Timestamp").min().alias("min_ts"),
        pl.col("Timestamp").max().alias("max_ts"),
    ).row(0)

    min_ts = float(timestamp_bounds[0]) if timestamp_bounds[0] is not None else 0.0
    max_ts = float(timestamp_bounds[1]) if timestamp_bounds[1] is not None else 1800.0

    slider_start = float(np.floor(min_ts))
    slider_stop = float(np.ceil(max_ts))
    if slider_stop <= slider_start:
        slider_stop = slider_start + 1.0
    return slider_start, slider_stop


@app.cell
def _(mo, slider_start, slider_stop, wells):
    selected_well_value = wells[0] if wells else None
    selected_well = mo.ui.dropdown(
        options=wells,
        value=selected_well_value,
        label="Selected well",
    )

    start_time = mo.ui.number(
        start=slider_start,
        stop=slider_stop,
        step=0.1,
        value=max(0.0, slider_start),
        label="Start time (s)",
    )
    end_time = mo.ui.number(
        start=slider_start,
        stop=slider_stop,
        step=0.1,
        value=min(120.0, slider_stop),
        label="End time (s)",
    )

    figure_width = mo.ui.number(start=4, stop=40, step=1, value=20, label="Figure width")
    figure_height = mo.ui.number(start=2, stop=20, step=0.5, value=5, label="Figure height")
    line_length = mo.ui.number(
        start=0.05, stop=2.0, step=0.05, value=0.8, label="Spike line length"
    )
    line_width = mo.ui.number(
        start=0.1, stop=4.0, step=0.1, value=0.6, label="Spike line width"
    )
    x_pad_left = mo.ui.number(start=0, stop=5, step=0.05, value=0.25, label="X padding left")
    x_pad_right = mo.ui.number(
        start=0, stop=5, step=0.05, value=1.25, label="X padding right"
    )
    spike_color = mo.ui.text(label="Spike color", value="black")
    show_channel_labels = mo.ui.checkbox(label="Show channel labels", value=False)
    return (
        end_time,
        figure_height,
        figure_width,
        line_length,
        line_width,
        selected_well,
        show_channel_labels,
        spike_color,
        start_time,
        x_pad_left,
        x_pad_right,
    )


@app.cell
def _(
    PlotSettings,
    end_time,
    figure_height,
    figure_width,
    line_length,
    line_width,
    show_channel_labels,
    spike_color,
    start_time,
    x_pad_left,
    x_pad_right,
):
    plot_settings = PlotSettings(
        start_time=float(start_time.value),
        end_time=float(end_time.value),
        figure_width=float(figure_width.value),
        figure_height=float(figure_height.value),
        line_length=float(line_length.value),
        line_width=float(line_width.value),
        color=spike_color.value.strip() or "black",
        x_pad_left=float(x_pad_left.value),
        x_pad_right=float(x_pad_right.value),
        show_channel_labels=bool(show_channel_labels.value),
    )
    return (plot_settings,)


@app.cell
def _(filter_well_window, plot_settings, rasterplot_data, selected_well):
    well_label = "" if selected_well.value is None else str(selected_well.value).strip()
    well_data = filter_well_window(
        rasterplot_data,
        well_label,
        plot_settings.start_time,
        plot_settings.end_time,
    )
    return well_data, well_label


@app.cell
def _(build_event_series, plot_settings, render_raster, well_data, well_label):
    events, channel_labels = build_event_series(well_data)
    fig = render_raster(events, channel_labels, well_label, plot_settings)
    return channel_labels, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plot
    """)
    return


@app.cell
def _(
    end_time,
    fig,
    figure_height,
    figure_width,
    line_length,
    line_width,
    mo,
    selected_well,
    show_channel_labels,
    spike_color,
    start_time,
    x_pad_left,
    x_pad_right,
):
    plot_control_widgets = [
        mo.md("### Plot Controls"),
        selected_well,
        mo.md("### Plot Window"),
        start_time,
        end_time,
        mo.md("### Plot Settings"),
        figure_width,
        figure_height,
        line_length,
        line_width,
        x_pad_left,
        x_pad_right,
        spike_color,
        show_channel_labels,
    ]
    controls = mo.vstack(
        plot_control_widgets,
        align="stretch",
        gap=0.3,
    )

    ax = fig.axes[0] if fig.axes else fig.gca()
    mo.vstack([controls, mo.ui.matplotlib(ax)], align="start", gap=1.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Summary
    """)
    return


@app.cell
def _(channel_labels, mo, spike_csv, well_data, well_label, wells):
    selected_well_label = well_label or "None"
    summary_spike_csv = spike_csv or ""
    mo.md(
        "\n".join(
            [
                "### Data Summary",
                f"- Spike CSV: `{summary_spike_csv}`",
                f"- Available wells in data: `{len(wells)}`",
                f"- Selected well: `{selected_well_label}`",
                f"- Spikes in current view: `{well_data.height}`",
                f"- Channels in current view: `{len(channel_labels)}`",
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
