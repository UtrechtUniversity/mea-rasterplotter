import marimo

__generated_with = "0.24.0"
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
    import numpy as np
    import polars as pl
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    SPIKE_REQUIRED_COLUMNS = {
        "Channel_Label",
        "Well_Label",
        "Timestamp",
    }
    PSTH_BIN_WIDTH_SECONDS = 0.001
    SMOOTHING_KERNEL_SAMPLE_COUNT = 1000

    @dataclass
    class PlotSettings:
        start_time: float
        end_time: float
        figure_width: float
        figure_height: float
        display_dpi: int
        line_length: float
        line_width: float
        spike_count_smoothing_method: str
        spike_count_exponential_tau_seconds: float
        spike_count_gaussian_width_factor: float
        spike_count_trace_line_width: float
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
        wells = (
            df.select(pl.col("Well_Label").drop_nulls().unique(maintain_order=True))
            .to_series()
            .to_list()
        )
        return [str(well) for well in wells]

    def combine_available_wells(*well_lists: list[str]) -> list[str]:
        well_labels: list[str] = []
        seen: set[str] = set()
        for well_list in well_lists:
            for well in well_list:
                well_label = str(well)
                if well_label not in seen:
                    well_labels.append(well_label)
                    seen.add(well_label)

        def _well_sort_key(well_label: str) -> tuple[str, float, str]:
            normalized_label = well_label.strip()
            row_label = normalized_label.rstrip("0123456789")
            column_label = normalized_label[len(row_label) :]
            column_number = (
                float(column_label) if column_label.isdigit() else float("inf")
            )
            return row_label.casefold(), column_number, normalized_label.casefold()

        return sorted(well_labels, key=_well_sort_key)

    def available_channels(df: pl.DataFrame) -> list[str]:
        channels = (
            df.select(
                pl.col("Channel_Label")
                .cast(pl.Utf8)
                .drop_nulls()
                .unique()
            )
            .to_series()
            .to_list()
        )
        return [str(channel) for channel in channels]

    def sorted_channel_union(*channel_lists: list[str]) -> list[str]:
        return sorted({str(channel) for channel_list in channel_lists for channel in channel_list})

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

    def nice_time_scale_seconds(window_seconds: float) -> float:
        if not np.isfinite(window_seconds) or window_seconds <= 0:
            return 0.0

        max_scale = 0.5 * window_seconds
        target_scale = 0.2 * window_seconds
        exponent_min = int(np.floor(np.log10(max_scale))) - 1
        exponent_max = int(np.ceil(np.log10(max_scale))) + 1
        candidates = sorted(
            {
                base * (10.0**exponent)
                for exponent in range(exponent_min, exponent_max + 1)
                for base in (1.0, 2.0, 5.0)
                if 0 < base * (10.0**exponent) <= max_scale
            }
        )
        if not candidates:
            return max_scale
        return min(candidates, key=lambda candidate: abs(candidate - target_scale))

    def format_time_scale_label(seconds: float) -> str:
        if seconds == 1:
            return "1 second"
        if float(seconds).is_integer():
            return f"{int(seconds)} seconds"
        return f"{seconds:g} seconds"

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

    def build_event_series(
        df: pl.DataFrame, channel_order: list[str]
    ) -> list[np.ndarray]:
        events: list[np.ndarray] = []

        for channel in channel_order:
            timestamps = (
                df.filter(pl.col("Channel_Label") == channel)
                .select("Timestamp")
                .to_series()
                .drop_nulls()
                .to_numpy()
            )
            events.append(np.asarray(timestamps, dtype=float))

        return events

    def gaussian_filter_counts(
        counts: np.ndarray,
        width_factor: float,
    ) -> np.ndarray:
        if width_factor <= 0:
            raise ValueError("width_factor must be positive")

        if counts.size == 0:
            return counts.astype(float, copy=False)

        # Match MATLAB gausswin(1000, alpha): the even-length window is sampled
        # symmetrically around the half-sample point. At alpha=2.5 its standard
        # deviation is 999 / (2 * 2.5) = 199.8 bins (199.8 ms on the fixed grid).
        offsets = (
            np.arange(SMOOTHING_KERNEL_SAMPLE_COUNT, dtype=float)
            - (SMOOTHING_KERNEL_SAMPLE_COUNT - 1) / 2.0
        )
        sigma_bins = (SMOOTHING_KERNEL_SAMPLE_COUNT - 1) / (2.0 * width_factor)
        kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
        kernel /= kernel.sum()

        # Reproduce MATLAB conv(counts, kernel, "same") for an even kernel:
        # its center lies half a sample before the current histogram-bin center.
        # np.convolve reverses the kernel during convolution: with 499 zeros
        # on the left, counts[i] multiplies kernel[500] at output[i].
        pad_left = (SMOOTHING_KERNEL_SAMPLE_COUNT - 1) // 2
        pad_right = SMOOTHING_KERNEL_SAMPLE_COUNT - 1 - pad_left
        padded_counts = np.pad(
            counts.astype(float, copy=False),
            (pad_left, pad_right),
            mode="constant",
        )
        return np.convolve(padded_counts, kernel, mode="valid")

    def exponential_filter_counts(
        counts: np.ndarray,
        tau_bins: float,
    ) -> np.ndarray:
        if tau_bins < 0:
            raise ValueError("tau_bins must be non-negative")

        # A zero time constant intentionally preserves the raw binned histogram.
        if tau_bins == 0 or counts.size == 0:
            return counts.astype(float, copy=False)

        # One-sided causal exponential with 1000 samples of support. At the fixed
        # 1 ms PSTH grid, the default 100 ms time constant gives one second of support.
        offsets = np.arange(SMOOTHING_KERNEL_SAMPLE_COUNT, dtype=float)
        kernel = np.exp(-offsets / tau_bins)
        kernel /= kernel.sum()

        # Full convolution followed by a left-aligned crop is causal: samples before
        # the displayed window are treated as zero and no future counts affect the trace.
        return np.convolve(
            counts.astype(float, copy=False), kernel, mode="full"
        )[: counts.size]

    def build_spike_rate_trace(
        df: pl.DataFrame,
        window_start: float,
        window_end: float,
        electrode_count: int,
        *,
        smoothing_method: str = "exponential",
        exponential_tau_seconds: float = 0.100,
        gaussian_width_factor: float = 2.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        if smoothing_method not in {"exponential", "gaussian"}:
            raise ValueError(f"Unknown smoothing method: {smoothing_method!r}")
        if exponential_tau_seconds < 0:
            raise ValueError("exponential_tau_seconds must be non-negative")
        if gaussian_width_factor <= 0:
            raise ValueError("gaussian_width_factor must be positive")

        bin_width_seconds = PSTH_BIN_WIDTH_SECONDS

        # Selected time span in seconds; clamp reversed/empty windows to a single bin:
        window_seconds = max(0.0, window_end - window_start)
        # Number of fixed-width histogram bins needed to cover the selected window:
        bin_count = max(1, int(np.ceil(window_seconds / bin_width_seconds)))
        # Absolute timestamp boundaries for np.histogram (not offsets from zero):
        bin_edges = window_start + np.arange(bin_count + 1, dtype=float) * bin_width_seconds
        if window_seconds > 0:
            # Clamp the final edge so the trace ends exactly at the requested window end:
            bin_edges[-1] = window_end

        timestamps = (
            df.select("Timestamp")
            .to_series()
            .drop_nulls()
            .to_numpy()
        )
        # np.histogram gives one spike count sum per interval. Smoothing changes
        # only the display trace and preserves the explicit histogram time grid.
        raw_spike_counts, _ = np.histogram(
            np.asarray(timestamps, dtype=float),
            bins=bin_edges,
        )
        if smoothing_method == "exponential":
            tau_bins = exponential_tau_seconds / bin_width_seconds
            smoothed_spike_counts = exponential_filter_counts(raw_spike_counts, tau_bins)
        else:
            smoothed_spike_counts = gaussian_filter_counts(
                raw_spike_counts, gaussian_width_factor
            )

        # Convert area-preserved counts to firing rate per electrode. The
        # shared channel list is the electrode denominator used in both plots.
        if electrode_count <= 0:
            spike_rates = np.zeros_like(smoothed_spike_counts, dtype=float)
        else:
            # Handle edge case where a clamped bin might not be exactly bin_width_seconds
            bin_durations = np.diff(bin_edges)
            spike_rates = smoothed_spike_counts / electrode_count / bin_durations
        return bin_edges, spike_rates

    def filter_well_for_plot(
        df: pl.DataFrame, well_label: str, settings: PlotSettings
    ) -> pl.DataFrame:
        return filter_well_window(
            df,
            well_label,
            settings.start_time,
            settings.end_time,
        )


    def make_eventplot_figure(
        events: list[np.ndarray],
        channel_labels: list[str],
        spike_bin_edges: np.ndarray,
        spike_rates: np.ndarray,
        well_label: str,
        settings: PlotSettings,
        title: str | None = None,
        *,
        show_time_axis: bool = False,
    ) -> Figure:
        window_start = min(settings.start_time, settings.end_time)
        window_end = max(settings.start_time, settings.end_time)
        window_seconds = window_end - window_start
        scale_seconds = nice_time_scale_seconds(window_seconds)
        fig = Figure(
            figsize=(settings.figure_width, settings.figure_height + (0.5 if show_time_axis else 0.0)),
            dpi=settings.display_dpi,
            constrained_layout=True,
        )
        trace_ax, ax, scale_ax = fig.subplots(
            nrows=3,
            sharex=True,
            gridspec_kw={"height_ratios": [0.4, 1.0, 0.1], "hspace": 0.04},
        )

        if len(spike_bin_edges) > 1 and len(spike_rates) > 0:
            # Get the center point of each bin (ndarray `+`` does element-wise addition):
            spike_bin_centers = 0.5 * (spike_bin_edges[:-1] + spike_bin_edges[1:])
            trace_ax.plot(
                spike_bin_centers,
                spike_rates,
                color=settings.color,
                linewidth=settings.spike_count_trace_line_width,
            )
        trace_ax.set_ylim(bottom=0)
        trace_ax.tick_params(axis="x", bottom=False, labelbottom=False)
        trace_ax.tick_params(axis="y", left=False, labelleft=False)
        trace_ax.grid(False)
        trace_ax.spines["top"].set_visible(False)
        trace_ax.spines["right"].set_visible(False)
        trace_ax.spines["bottom"].set_visible(False)
        trace_ax.spines["left"].set_visible(False)

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
                ax.tick_params(axis="y", length=0)
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
        ax.tick_params(axis="x", bottom=show_time_axis, labelbottom=show_time_axis)
        if show_time_axis:
            ax.set_xlabel("Time (s)")
            locator = MaxNLocator(nbins=max(2, int(settings.figure_width)))
            ticks = np.asarray(locator.tick_values(window_start, window_end), dtype=float)
            ticks = ticks[(ticks >= window_start) & (ticks <= window_end)]
            if window_start == window_end:
                ticks = np.array([window_start])
            ax.set_xticks(ticks)
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.xaxis.set_major_formatter(formatter)
        ax.set_ylabel("")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_bounds(window_start, window_end)

        scale_ax.set_ylim(0, 1)
        scale_ax.tick_params(
            axis="both",
            left=False,
            labelleft=False,
            bottom=False,
            labelbottom=False,
        )
        scale_ax.grid(False)
        for spine in scale_ax.spines.values():
            spine.set_visible(False)
        if scale_seconds > 0:
            scale_end = window_end
            scale_start = scale_end - scale_seconds
            scale_ax.plot(
                [scale_start, scale_end],
                [0.25, 0.25],
                color="black",
                linewidth=3,
                solid_capstyle="butt",
                clip_on=False,
            )
            scale_ax.text(
                scale_start + scale_seconds / 2,
                0.4,
                format_time_scale_label(scale_seconds),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        return fig

    return (
        Path,
        PlotSettings,
        available_channels,
        available_wells,
        build_event_series,
        build_spike_rate_trace,
        combine_available_wells,
        filter_well_for_plot,
        load_spike_csv,
        make_eventplot_figure,
        mo,
        sorted_channel_union,
        timestamp_slider_bounds,
    )


@app.cell
def _(build_event_series, build_spike_rate_trace, make_eventplot_figure):
    def make_spike_raster_figures(
        dataset_label: str,
        well_spikes,
        channel_labels: list[str],
        well_label: str,
        settings,
    ):
        """Build a time-labelled preview and the original download from shared data."""
        window_start = min(settings.start_time, settings.end_time)
        window_end = max(settings.start_time, settings.end_time)
        events = build_event_series(well_spikes, channel_labels)
        bin_edges, rates = build_spike_rate_trace(
            well_spikes,
            window_start,
            window_end,
            len(channel_labels),
            smoothing_method=settings.spike_count_smoothing_method,
            exponential_tau_seconds=settings.spike_count_exponential_tau_seconds,
            gaussian_width_factor=settings.spike_count_gaussian_width_factor,
        )
        display_fig, download_fig = (
            make_eventplot_figure(
                events, channel_labels, bin_edges, rates, well_label, settings,
                title=f"{dataset_label} Raster Plot for Well {well_label}",
                show_time_axis=show_time_axis,
            )
            for show_time_axis in (True, False)
        )
        return display_fig, download_fig


    return (make_spike_raster_figures,)


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
    baseline_csv_picker = mo.ui.file_browser(
        initial_path=initial_csv_dir,
        filetypes=[".csv"],
        multiple=False,
        restrict_navigation=False,
        label="Baseline CSV file",
        on_change=lambda _: set_show_csv_pickers(False),
    )
    exposure_csv_picker = mo.ui.file_browser(
        initial_path=initial_csv_dir,
        filetypes=[".csv"],
        multiple=False,
        restrict_navigation=False,
        label="Exposure CSV file",
        on_change=lambda _: set_show_csv_pickers(False),
    )
    return baseline_csv_picker, csv_picker_toggle, exposure_csv_picker


@app.cell
def _(
    baseline_csv_picker,
    csv_picker_toggle,
    exposure_csv_picker,
    mo,
    show_csv_pickers,
):
    displayed_baseline_csv = baseline_csv_picker.path(0) or ""
    displayed_exposure_csv = exposure_csv_picker.path(0) or ""
    input_widgets = [
        mo.md("### Inputs"),
        mo.md(f"Baseline CSV: `{displayed_baseline_csv or 'None selected'}`"),
        mo.md(f"Exposure CSV: `{displayed_exposure_csv or 'None selected'}`"),
        csv_picker_toggle,
    ]
    if show_csv_pickers() or not (displayed_baseline_csv and displayed_exposure_csv):
        input_widgets.extend([baseline_csv_picker, exposure_csv_picker])
    if not displayed_baseline_csv or not displayed_exposure_csv:
        input_widgets.append(
            mo.md("Select both a baseline CSV and an exposure CSV to continue.").callout(kind="warn")
        )
    mo.vstack(input_widgets, align="stretch", gap=0.3)
    return


@app.cell
def _(Path, baseline_csv_picker, exposure_csv_picker, mo):
    selected_baseline_csv = baseline_csv_picker.path(0)
    selected_exposure_csv = exposure_csv_picker.path(0)
    resolved_baseline_csv = Path(selected_baseline_csv) if selected_baseline_csv else None
    resolved_exposure_csv = Path(selected_exposure_csv) if selected_exposure_csv else None
    mo.stop(resolved_baseline_csv is None or resolved_exposure_csv is None)
    return resolved_baseline_csv, resolved_exposure_csv


@app.cell
def _(load_spike_csv, mo, resolved_baseline_csv, resolved_exposure_csv):
    mo.stop(resolved_baseline_csv is None or resolved_exposure_csv is None)
    baseline_data = load_spike_csv(resolved_baseline_csv, "Baseline CSV")
    exposure_data = load_spike_csv(resolved_exposure_csv, "Exposure CSV")
    return baseline_data, exposure_data


@app.cell
def _(
    available_channels,
    available_wells,
    baseline_data,
    combine_available_wells,
    exposure_data,
    sorted_channel_union,
):
    baseline_wells = available_wells(baseline_data)
    exposure_wells = available_wells(exposure_data)
    well_labels = combine_available_wells(baseline_wells, exposure_wells)
    _baseline_channel_labels = available_channels(baseline_data)
    _exposure_channel_labels = available_channels(exposure_data)
    shared_channel_labels = sorted_channel_union(
        _baseline_channel_labels,
        _exposure_channel_labels,
    )
    return shared_channel_labels, well_labels


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
    well_labels,
):
    selected_well_value = well_labels[0] if well_labels else None
    selected_well = mo.ui.dropdown(
        options=well_labels,
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

    figure_width = mo.ui.number(start=4, stop=40, step=1, value=6, label="Figure width (in)")
    figure_height = mo.ui.number(start=2, stop=20, step=0.5, value=4, label="Figure height (in)")
    display_dpi = mo.ui.dropdown(
        options={
            "72": 72,
            "100": 100,
            "150": 150,
            "200": 200,
            "300": 300,
            "600": 600,
        },
        value="100",
        label="Display DPI",
    )
    download_dpi = mo.ui.dropdown(
        options={
            "72": 72,
            "100": 100,
            "150": 150,
            "200": 200,
            "300": 300,
            "600": 600,
        },
        value="300",
        label="Download DPI",
    )
    line_length = mo.ui.number(
        start=0.05, stop=2.0, step=0.05, value=0.8, label="Spike line length (y-axis units)"
    )
    line_width = mo.ui.number(
        start=0.1, stop=4.0, step=0.1, value=0.6, label="Spike line width (pt)"
    )
    spike_count_bin_width_ms = mo.ui.number(
        start=1,
        stop=1000,
        step=1,
        value=1,
        label="Histogram bin width (ms, fixed)",
        disabled=True,
    )
    spike_count_smoothing_method = mo.ui.dropdown(
        options={"Exponential decay": "exponential", "Gaussian": "gaussian"},
        value="Exponential decay",
        label="Smoothing method",
    )
    get_exponential_tau_ms, set_exponential_tau_ms = mo.state(100)
    get_gaussian_width_factor, set_gaussian_width_factor = mo.state(2.5)
    spike_count_trace_line_width = mo.ui.number(
        start=0.1, stop=4.0, step=0.1, value=0.2, label="Histogram line width (pt)"
    )
    x_pad_left = mo.ui.number(start=0, stop=5, step=0.05, value=1, label="X padding left (s)")
    x_pad_right = mo.ui.number(
        start=0, stop=5, step=0.05, value=0, label="X padding right (s)"
    )
    spike_color = mo.ui.dropdown(
        options={
            "Black": "black",
            "Dark gray": "0.25",
            "Blue": "tab:blue",
            "Orange": "tab:orange",
            "Green": "tab:green",
            "Red": "tab:red",
            "Purple": "tab:purple",
        },
        value="Black",
        label="Spike color",
    )
    show_channel_labels = mo.ui.checkbox(label="Show channel labels", value=True)
    return (
        baseline_end_time,
        baseline_start_time,
        display_dpi,
        download_dpi,
        exposure_end_time,
        exposure_start_time,
        figure_height,
        figure_width,
        get_exponential_tau_ms,
        get_gaussian_width_factor,
        line_length,
        line_width,
        selected_well,
        set_exponential_tau_ms,
        set_gaussian_width_factor,
        show_channel_labels,
        spike_color,
        spike_count_bin_width_ms,
        spike_count_smoothing_method,
        spike_count_trace_line_width,
        x_pad_left,
        x_pad_right,
    )


@app.cell
def _(
    PlotSettings,
    baseline_end_time,
    baseline_start_time,
    display_dpi,
    exposure_end_time,
    exposure_start_time,
    figure_height,
    figure_width,
    line_length,
    line_width,
    show_channel_labels,
    spike_color,
    spike_count_exponential_tau_ms,
    spike_count_gaussian_width_factor,
    spike_count_smoothing_method,
    spike_count_trace_line_width,
    x_pad_left,
    x_pad_right,
):
    def make_plot_settings(start_widget, end_widget):
        return PlotSettings(
            start_time=float(start_widget.value),
            end_time=float(end_widget.value),
            figure_width=float(figure_width.value),
            figure_height=float(figure_height.value),
            display_dpi=int(display_dpi.value),
            line_length=float(line_length.value),
            line_width=float(line_width.value),
            spike_count_smoothing_method=spike_count_smoothing_method.value,
            spike_count_exponential_tau_seconds=float(spike_count_exponential_tau_ms.value) / 1000.0,
            spike_count_gaussian_width_factor=float(spike_count_gaussian_width_factor.value),
            spike_count_trace_line_width=float(spike_count_trace_line_width.value),
            color=spike_color.value,
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
    filter_well_for_plot,
    selected_well,
):
    well_label = "" if selected_well.value is None else str(selected_well.value).strip()
    baseline_well_data = filter_well_for_plot(
        baseline_data,
        well_label,
        baseline_plot_settings,
    )
    exposure_well_data = filter_well_for_plot(
        exposure_data,
        well_label,
        exposure_plot_settings,
    )
    return baseline_well_data, exposure_well_data, well_label


@app.cell
def _(
    baseline_plot_settings,
    baseline_well_data,
    exposure_plot_settings,
    exposure_well_data,
    make_spike_raster_figures,
    shared_channel_labels,
    well_label,
):
    baseline_display_fig, baseline_fig = make_spike_raster_figures(
        "Baseline",
        baseline_well_data,
        shared_channel_labels,
        well_label,
        baseline_plot_settings,
    )
    exposure_display_fig, exposure_fig = make_spike_raster_figures(
        "Exposure",
        exposure_well_data,
        shared_channel_labels,
        well_label,
        exposure_plot_settings,
    )
    return (
        baseline_display_fig,
        baseline_fig,
        exposure_display_fig,
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
    display_dpi,
    download_dpi,
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
    spike_count_bin_width_ms,
    spike_count_smoothing_method,
    spike_count_smoothing_parameter,
    spike_count_trace_line_width,
    x_pad_left,
):
    figure_settings_column = mo.vstack(
        [
            mo.md("### Figure"),
            figure_width,
            figure_height,
            mo.md("Figure height sets the download height; live previews add space for the time axis."),
            display_dpi,
            download_dpi,
            x_pad_left,
            spike_color,
        ],
        align="stretch",
        gap=0.3,
    )
    raster_settings_column = mo.vstack(
        [
            mo.md("### Raster plot"),
            line_length,
            line_width,
            show_channel_labels
        ],
        align="stretch",
        gap=0.3,
    )
    trace_settings_column = mo.vstack(
        [
            mo.md("### Population spike time histogram"),
            spike_count_bin_width_ms,
            spike_count_smoothing_method,
            spike_count_smoothing_parameter,
            spike_count_trace_line_width,
        ],
        align="stretch",
        gap=0.3,
    )
    shared_plot_settings = mo.accordion(
        {
            "Plot Settings": mo.hstack(
                [
                    figure_settings_column,
                    raster_settings_column,
                    trace_settings_column,
                ],
                widths="equal",
                align="start",
                wrap=True,
                gap=1.0,
            )
        }
    )
    baseline_time_window_panel = mo.vstack(
        [
            mo.md("### Baseline Time Window"),
            baseline_start_time,
            baseline_end_time,
        ],
        align="stretch",
        gap=0.3,
    )
    exposure_time_window_panel = mo.vstack(
        [
            mo.md("### Exposure Time Window"),
            exposure_start_time,
            exposure_end_time,
        ],
        align="stretch",
        gap=0.3,
    )
    time_window_controls = mo.hstack(
        [baseline_time_window_panel, exposure_time_window_panel],
        widths="equal",
        align="start",
        wrap=True,
        gap=1.0,
    )
    plot_control_widgets = [
        shared_plot_settings,
        mo.md("### Plot Controls"),
        selected_well,
        time_window_controls,
    ]
    mo.vstack(
        plot_control_widgets,
        align="stretch",
        gap=0.3,
    )
    return


@app.cell(hide_code=True)
def _(
    baseline_display_fig,
    baseline_fig,
    baseline_plot_settings,
    download_dpi,
    exposure_display_fig,
    exposure_fig,
    exposure_plot_settings,
    mo,
    well_label,
):
    import io
    import re


    def _figure_png_bytes(fig, dpi: int) -> bytes:
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        return buffer.getvalue()


    def _download_filename(dataset_label: str, well_label: str, settings, dpi: int) -> str:
        safe_well = re.sub(r"[^A-Za-z0-9_.-]+", "-", well_label or "none").strip("-")
        start_time = min(settings.start_time, settings.end_time)
        end_time = max(settings.start_time, settings.end_time)
        return (
            f"{dataset_label.lower()}_well-{safe_well}_"
            f"{start_time:g}-{end_time:g}s_{dpi}dpi.png"
        )


    def _plot_download(fig, dataset_label: str, well_label: str, settings):
        dpi = int(download_dpi.value)
        return mo.download(
            data=lambda: _figure_png_bytes(fig, dpi),
            filename=_download_filename(dataset_label, well_label, settings, dpi),
            mimetype="image/png",
            label=f"Download {dataset_label.lower()} PNG",
        )


    baseline_panel = mo.vstack(
        [
            mo.md("### Baseline"),
            baseline_display_fig,
            _plot_download(baseline_fig, "Baseline", well_label, baseline_plot_settings),
        ],
        align="start",
        gap=0.5,
    )
    exposure_panel = mo.vstack(
        [
            mo.md("### Exposure"),
            exposure_display_fig,
            _plot_download(exposure_fig, "Exposure", well_label, exposure_plot_settings),
        ],
        align="start",
        gap=0.5,
    )

    mo.hstack(
        [baseline_panel, exposure_panel],
        widths="equal",
        align="start",
        wrap=True,
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
    baseline_well_data,
    exposure_well_data,
    mo,
    shared_channel_labels,
    well_labels,
):
    mo.md(
        "\n".join(
            [
                f"- Available wells across both files: `{len(well_labels)}`",
                f"- Baseline spikes in current view: `{baseline_well_data.height}`",
                f"- Exposure spikes in current view: `{exposure_well_data.height}`",
                f"- Shared channels shown: `{len(shared_channel_labels)}`",
            ]
        )
    )
    return


@app.cell
def _(
    get_exponential_tau_ms,
    get_gaussian_width_factor,
    mo,
    set_exponential_tau_ms,
    set_gaussian_width_factor,
    spike_count_smoothing_method,
):
    _use_exponential = spike_count_smoothing_method.value == "exponential"
    spike_count_exponential_tau_ms = mo.ui.number(
        start=0,
        stop=1000,
        step=1,
        value=get_exponential_tau_ms(),
        label="Exponential time constant (ms)",
        disabled=not _use_exponential,
        on_change=set_exponential_tau_ms,
    )
    spike_count_gaussian_width_factor = mo.ui.number(
        start=0.1,
        stop=100,
        step=0.1,
        value=get_gaussian_width_factor(),
        label="Gaussian width factor",
        disabled=_use_exponential,
        on_change=set_gaussian_width_factor,
    )
    spike_count_smoothing_parameter = (
        spike_count_exponential_tau_ms
        if _use_exponential
        else spike_count_gaussian_width_factor
    )
    return (
        spike_count_exponential_tau_ms,
        spike_count_gaussian_width_factor,
        spike_count_smoothing_parameter,
    )


if __name__ == "__main__":
    app.run()
