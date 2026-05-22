import marimo

__generated_with = "0.23.7"
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

    def load_spike_csv(path: Path, dataset_name: str = "Spike CSV") -> pl.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(f"{dataset_name} not found: {path}")
        spikes = pl.read_csv(path)
        assert_required_columns(spikes, SPIKE_REQUIRED_COLUMNS, dataset_name)
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

    def combine_available_wells(*well_lists: list[str]) -> list[str]:
        wells: list[str] = []
        seen: set[str] = set()
        for well_list in well_lists:
            for well in well_list:
                well_label = str(well)
                if well_label not in seen:
                    wells.append(well_label)
                    seen.add(well_label)
        return wells

    def timestamp_slider_bounds(df: pl.DataFrame) -> tuple[float, float]:
        timestamp_bounds = df.select(
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
        title: str | None = None,
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
        ax.set_title(title or f"Raster Plot for Well {well_label}")
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
        combine_available_wells,
        filter_well_window,
        load_spike_csv,
        mo,
        render_raster,
        timestamp_slider_bounds,
    )


@app.cell
def _(Path):
    notebook_dir = Path(__file__).resolve().parent
    initial_csv_dir = notebook_dir
    return (initial_csv_dir,)


@app.cell
def _(mo):
    show_csv_pickers, set_show_csv_pickers = mo.state(True)
    return set_show_csv_pickers, show_csv_pickers


@app.cell
def _(initial_csv_dir, mo, set_show_csv_pickers):
    csv_picker_toggle = mo.ui.button(
        label="Choose/change CSV files",
        on_click=lambda _: set_show_csv_pickers(lambda current: not current),
    )
    baseline_csv_path = mo.ui.file_browser(
        initial_path=initial_csv_dir,
        filetypes=[".csv"],
        multiple=False,
        label="Baseline CSV file",
        on_change=lambda _: set_show_csv_pickers(False),
    )
    exposure_csv_path = mo.ui.file_browser(
        initial_path=initial_csv_dir,
        filetypes=[".csv"],
        multiple=False,
        label="Exposure CSV file",
        on_change=lambda _: set_show_csv_pickers(False),
    )
    return baseline_csv_path, csv_picker_toggle, exposure_csv_path


@app.cell
def _(
    baseline_csv_path,
    csv_picker_toggle,
    exposure_csv_path,
    mo,
    show_csv_pickers,
):
    displayed_baseline_csv = baseline_csv_path.path(0) or ""
    displayed_exposure_csv = exposure_csv_path.path(0) or ""
    input_widgets = [
        mo.md("### Inputs"),
        mo.md(f"Baseline CSV: `{displayed_baseline_csv or 'None selected'}`"),
        mo.md(f"Exposure CSV: `{displayed_exposure_csv or 'None selected'}`"),
        csv_picker_toggle,
    ]
    if show_csv_pickers() or not (displayed_baseline_csv and displayed_exposure_csv):
        input_widgets.extend([baseline_csv_path, exposure_csv_path])
    if not displayed_baseline_csv or not displayed_exposure_csv:
        input_widgets.append(
            mo.md("Select both a baseline CSV and an exposure CSV to continue.").callout(kind="warn")
        )
    mo.vstack(input_widgets, align="stretch", gap=0.3)
    return


@app.cell
def _(Path, baseline_csv_path, exposure_csv_path, mo):
    selected_baseline_csv_path = baseline_csv_path.path(0)
    selected_exposure_csv_path = exposure_csv_path.path(0)
    resolved_baseline_csv = Path(selected_baseline_csv_path) if selected_baseline_csv_path else None
    resolved_exposure_csv = Path(selected_exposure_csv_path) if selected_exposure_csv_path else None
    mo.stop(resolved_baseline_csv is None or resolved_exposure_csv is None)
    return resolved_baseline_csv, resolved_exposure_csv


@app.cell
def _(load_spike_csv, mo, resolved_baseline_csv, resolved_exposure_csv):
    mo.stop(resolved_baseline_csv is None or resolved_exposure_csv is None)
    baseline_data = load_spike_csv(resolved_baseline_csv, "Baseline CSV")
    exposure_data = load_spike_csv(resolved_exposure_csv, "Exposure CSV")
    baseline_csv = resolved_baseline_csv
    exposure_csv = resolved_exposure_csv
    return baseline_csv, baseline_data, exposure_csv, exposure_data


@app.cell
def _(available_wells, baseline_data, combine_available_wells, exposure_data):
    baseline_wells = available_wells(baseline_data)
    exposure_wells = available_wells(exposure_data)
    wells = combine_available_wells(baseline_wells, exposure_wells)
    return (wells,)


@app.cell
def _(baseline_data, exposure_data, timestamp_slider_bounds):
    baseline_slider_start, baseline_slider_stop = timestamp_slider_bounds(baseline_data)
    exposure_slider_start, exposure_slider_stop = timestamp_slider_bounds(exposure_data)
    return (
        baseline_slider_start,
        baseline_slider_stop,
        exposure_slider_start,
        exposure_slider_stop,
    )


@app.cell
def _(
    baseline_slider_start,
    baseline_slider_stop,
    exposure_slider_start,
    exposure_slider_stop,
    mo,
    wells,
):
    selected_well_value = wells[0] if wells else None
    selected_well = mo.ui.dropdown(
        options=wells,
        value=selected_well_value,
        label="Selected well",
    )

    baseline_start_time = mo.ui.number(
        start=baseline_slider_start,
        stop=baseline_slider_stop,
        step=0.1,
        value=max(0.0, baseline_slider_start),
        label="Baseline start time (s)",
    )
    baseline_end_time = mo.ui.number(
        start=baseline_slider_start,
        stop=baseline_slider_stop,
        step=0.1,
        value=min(120.0, baseline_slider_stop),
        label="Baseline end time (s)",
    )
    exposure_start_time = mo.ui.number(
        start=exposure_slider_start,
        stop=exposure_slider_stop,
        step=0.1,
        value=max(0.0, exposure_slider_start),
        label="Exposure start time (s)",
    )
    exposure_end_time = mo.ui.number(
        start=exposure_slider_start,
        stop=exposure_slider_stop,
        step=0.1,
        value=min(120.0, exposure_slider_stop),
        label="Exposure end time (s)",
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
        baseline_end_time,
        baseline_start_time,
        exposure_end_time,
        exposure_start_time,
        figure_height,
        figure_width,
        line_length,
        line_width,
        selected_well,
        show_channel_labels,
        spike_color,
        x_pad_left,
        x_pad_right,
    )


@app.cell
def _(
    PlotSettings,
    baseline_end_time,
    baseline_start_time,
    exposure_end_time,
    exposure_start_time,
    figure_height,
    figure_width,
    line_length,
    line_width,
    show_channel_labels,
    spike_color,
    x_pad_left,
    x_pad_right,
):
    def make_plot_settings(start_widget, end_widget) -> PlotSettings:
        return PlotSettings(
            start_time=float(start_widget.value),
            end_time=float(end_widget.value),
            figure_width=float(figure_width.value),
            figure_height=float(figure_height.value),
            line_length=float(line_length.value),
            line_width=float(line_width.value),
            color=spike_color.value.strip() or "black",
            x_pad_left=float(x_pad_left.value),
            x_pad_right=float(x_pad_right.value),
            show_channel_labels=bool(show_channel_labels.value),
        )

    baseline_plot_settings = make_plot_settings(baseline_start_time, baseline_end_time)
    exposure_plot_settings = make_plot_settings(exposure_start_time, exposure_end_time)
    return baseline_plot_settings, exposure_plot_settings


@app.cell
def _(
    baseline_data,
    baseline_plot_settings,
    exposure_data,
    exposure_plot_settings,
    filter_well_window,
    selected_well,
):
    well_label = "" if selected_well.value is None else str(selected_well.value).strip()
    baseline_well_data = filter_well_window(
        baseline_data,
        well_label,
        baseline_plot_settings.start_time,
        baseline_plot_settings.end_time,
    )
    exposure_well_data = filter_well_window(
        exposure_data,
        well_label,
        exposure_plot_settings.start_time,
        exposure_plot_settings.end_time,
    )
    return baseline_well_data, exposure_well_data, well_label


@app.cell
def _(
    baseline_plot_settings,
    baseline_well_data,
    build_event_series,
    exposure_plot_settings,
    exposure_well_data,
    render_raster,
    well_label,
):
    baseline_events, baseline_channel_labels = build_event_series(baseline_well_data)
    exposure_events, exposure_channel_labels = build_event_series(exposure_well_data)
    baseline_fig = render_raster(
        baseline_events,
        baseline_channel_labels,
        well_label,
        baseline_plot_settings,
        title=f"Baseline Raster Plot for Well {well_label}",
    )
    exposure_fig = render_raster(
        exposure_events,
        exposure_channel_labels,
        well_label,
        exposure_plot_settings,
        title=f"Exposure Raster Plot for Well {well_label}",
    )
    return (
        baseline_channel_labels,
        baseline_fig,
        exposure_channel_labels,
        exposure_fig,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plot
    """)
    return


@app.cell
def _(
    baseline_end_time,
    baseline_start_time,
    exposure_end_time,
    exposure_start_time,
    figure_height,
    figure_width,
    line_length,
    line_width,
    mo,
    selected_well,
    show_channel_labels,
    spike_color,
    x_pad_left,
    x_pad_right,
):
    shared_plot_settings = mo.accordion(
        {
            "Plot Settings": mo.vstack(
                [
                    figure_width,
                    figure_height,
                    line_length,
                    line_width,
                    x_pad_left,
                    x_pad_right,
                    spike_color,
                ],
                align="stretch",
                gap=0.3,
            )
        }
    )
    plot_control_widgets = [
        shared_plot_settings,
        mo.md("### Plot Controls"),
        selected_well,
        show_channel_labels,
        mo.md("### Baseline Time Window"),
        baseline_start_time,
        baseline_end_time,
        mo.md("### Exposure Time Window"),
        exposure_start_time,
        exposure_end_time,
    ]
    mo.vstack(
        plot_control_widgets,
        align="stretch",
        gap=0.3,
    )
    return


@app.cell(hide_code=True)
def _(baseline_fig, exposure_fig, mo):
    baseline_ax = baseline_fig.axes[0] if baseline_fig.axes else baseline_fig.gca()
    exposure_ax = exposure_fig.axes[0] if exposure_fig.axes else exposure_fig.gca()
    mo.vstack(
        [
            mo.md("### Baseline"),
            mo.ui.matplotlib(baseline_ax),
            mo.md("### Exposure"),
            mo.ui.matplotlib(exposure_ax),
        ],
        align="start",
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Summary
    """)
    return


@app.cell
def _(
    baseline_channel_labels,
    baseline_csv,
    baseline_well_data,
    exposure_channel_labels,
    exposure_csv,
    exposure_well_data,
    mo,
    well_label,
    wells,
):
    selected_well_label = well_label or "None"
    summary_baseline_csv = baseline_csv or ""
    summary_exposure_csv = exposure_csv or ""
    mo.md(
        "\n".join(
            [
                "### Data Summary",
                f"- Baseline CSV: `{summary_baseline_csv}`",
                f"- Exposure CSV: `{summary_exposure_csv}`",
                f"- Available wells across both files: `{len(wells)}`",
                f"- Selected well: `{selected_well_label}`",
                f"- Baseline spikes in current view: `{baseline_well_data.height}`",
                f"- Baseline channels in current view: `{len(baseline_channel_labels)}`",
                f"- Exposure spikes in current view: `{exposure_well_data.height}`",
                f"- Exposure channels in current view: `{len(exposure_channel_labels)}`",
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
