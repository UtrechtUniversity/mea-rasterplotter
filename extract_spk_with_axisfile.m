function extract_spk_with_axisfile(spk_path, output_csv, loader_dir)
%EXTRACT_SPK_WITH_AXISFILE Convert one Axion .spk file to CSV using AxisFile.
%   extract_spk_with_axisfile(spk_path, output_csv, loader_dir)
%
%   - spk_path: path to input .spk file
%   - output_csv: path to output .csv file (optional; defaults to spk basename)
%   - loader_dir: path to AxionFileLoader class files (optional)

    if nargin < 1 || isempty(spk_path)
        error('extract_spk_with_axisfile:MissingInput', 'spk_path is required.');
    end

    if nargin < 2 || isempty(output_csv)
        [spk_parent, spk_name, ~] = fileparts(spk_path);
        output_csv = fullfile(spk_parent, [spk_name '.csv']);
    end

    if nargin < 3 || isempty(loader_dir)
        this_dir = fileparts(mfilename('fullpath'));
        loader_dir = fullfile(this_dir, '..', 'vendor', 'AxionFileLoader', 'AxionFileLoader');
    end

    if exist(spk_path, 'file') ~= 2
        error('extract_spk_with_axisfile:MissingFile', 'SPK file not found: %s', spk_path);
    end

    if exist(loader_dir, 'dir') ~= 7
        error('extract_spk_with_axisfile:MissingLoader', 'AxionFileLoader directory not found: %s', loader_dir);
    end

    addpath(loader_dir);

    out_parent = fileparts(output_csv);
    if ~isempty(out_parent) && exist(out_parent, 'dir') ~= 7
        mkdir(out_parent);
    end

    spike_dataset = AxisFile(spk_path).SpikeData;
    spike_rows = struct( ...
        'WellRow', [], ...
        'WellColumn', [], ...
        'ElectrodeColumn', [], ...
        'ElectrodeRow', [], ...
        'WaveformStartTime', [], ...
        'SpikeTime', [], ...
        'MaximumAmplitude', [], ...
        'MinimumAmplitude', [], ...
        'PeakToPeakAmplitude', []);

    if ~isempty(spike_dataset)
        for ds_idx = 1:numel(spike_dataset)
            current_rows = spike_dataset(ds_idx).LoadAllSpikesDetailed();
            if isempty(spike_rows.WaveformStartTime)
                spike_rows = current_rows;
            elseif ~isempty(current_rows.WaveformStartTime)
                spike_rows.WellRow = [spike_rows.WellRow, current_rows.WellRow];
                spike_rows.WellColumn = [spike_rows.WellColumn, current_rows.WellColumn];
                spike_rows.ElectrodeColumn = [spike_rows.ElectrodeColumn, current_rows.ElectrodeColumn];
                spike_rows.ElectrodeRow = [spike_rows.ElectrodeRow, current_rows.ElectrodeRow];
                spike_rows.WaveformStartTime = [spike_rows.WaveformStartTime, current_rows.WaveformStartTime];
                spike_rows.SpikeTime = [spike_rows.SpikeTime, current_rows.SpikeTime];
                spike_rows.MaximumAmplitude = [spike_rows.MaximumAmplitude, current_rows.MaximumAmplitude];
                spike_rows.MinimumAmplitude = [spike_rows.MinimumAmplitude, current_rows.MinimumAmplitude];
                spike_rows.PeakToPeakAmplitude = [spike_rows.PeakToPeakAmplitude, current_rows.PeakToPeakAmplitude];
            end
        end
    end

    final_results = [];
    if ~isempty(spike_rows.WaveformStartTime)
        electrode_digits = floor(log10(spike_rows.ElectrodeRow)) + 1;
        well_digits = floor(log10(spike_rows.WellColumn)) + 1;

        channel_label = spike_rows.ElectrodeColumn .* (10 .^ electrode_digits) + spike_rows.ElectrodeRow;
        well_label = spike_rows.WellRow .* (10 .^ well_digits) + spike_rows.WellColumn;

        % Keep the CSV headers aligned with the example: use "timestamp"
        timestamp = spike_rows.WaveformStartTime;
        final_results = [ ...
            channel_label(:), ...
            well_label(:), ...
            timestamp(:), ...
            spike_rows.MaximumAmplitude(:), ...
            spike_rows.MinimumAmplitude(:), ...
            spike_rows.PeakToPeakAmplitude(:) ...
        ];
    end

    fid = fopen(output_csv, 'w');
    if fid < 0
        error('extract_spk_with_axisfile:OpenOutput', 'Unable to open output CSV: %s', output_csv);
    end

    fprintf(fid, '%s,%s,%s,%s,%s,%s\n', ...
        'Channel_Label', ...
        'Well_Label', ...
        'Timestamp', ...
        'Maximum_Amplitude', ...
        'Minimum_Amplitude', ...
        'Peak_to_peak_Amplitude');
    fclose(fid);

    if ~isempty(final_results)
        dlmwrite(output_csv, final_results, '-append', 'delimiter', ',', 'precision', 17);
    end

    fprintf('Wrote CSV: %s\n', output_csv);
end
