function output_csv_path = extract_spk_with_axisfile_matlab(spk_path, output_csv, loader_dir)
%EXTRACT_SPK_WITH_AXISFILE_MATLAB Extract spike timings from one Axion .spk file.
%   output_csv_path = extract_spk_with_axisfile_matlab(spk_path, output_csv, loader_dir)
%   loads spike data with the AxionFileLoader MATLAB API,
%   maps hardware channels to well/electrode coordinates, and writes
%   trigger-aligned spike timings to a .csv file.
%
%   - spk_path: path to input .spk file
%   - output_csv: path to output .csv file (optional; defaults to spk basename)
%   - loader_dir: path to AxionFileLoader class files (optional)
%
%   Progress is written to stdout so callers can redirect it to a log file.

    if nargin < 1 || isempty(spk_path)
        error('extract_spk_with_axisfile_matlab:MissingInput', 'spk_path is required.');
    end

    if isstring(spk_path)
        spk_path = char(spk_path);
    end

    if ~(ischar(spk_path) && ~isempty(spk_path))
        error('extract_spk_with_axisfile_matlab:InvalidPath', ...
            'spk_path must be a non-empty character vector or string scalar.');
    end

    if nargin < 2 || isempty(output_csv)
        [spk_parent, spk_name, ~] = fileparts(spk_path);
        output_csv_path = fullfile(spk_parent, [spk_name '.csv']);
    else
        output_csv_path = output_csv;
        if isstring(output_csv_path)
            output_csv_path = char(output_csv_path);
        end
    end

    if nargin < 3 || isempty(loader_dir)
        script_dir = fileparts(mfilename('fullpath'));
        loader_dir = fullfile(script_dir, 'vendor', 'AxionFileLoader', 'AxionFileLoader');
    elseif isstring(loader_dir)
        loader_dir = char(loader_dir);
    end

    if exist(loader_dir, 'dir') ~= 7
        error('extract_spk_with_axisfile_matlab:MissingLoader', ...
            'AxionFileLoader directory not found: %s', loader_dir);
    end

    run_timer = tic;
    addpath(loader_dir);
    log_progress('Starting MATLAB SPK extraction.');
    log_progress(sprintf('Input SPK: %s', spk_path));
    log_progress(sprintf('Output CSV: %s', output_csv_path));
    log_progress(sprintf('AxionFileLoader directory: %s', loader_dir));

    if exist(spk_path, 'file') ~= 2
        error('extract_spk_with_axisfile_matlab:MissingFile', ...
            'SPK file not found: %s', spk_path);
    end

    [output_parent, ~, ~] = fileparts(output_csv_path);
    if ~isempty(output_parent) && exist(output_parent, 'dir') ~= 7
        mkdir(output_parent);
    end

    [~, ~, spk_ext] = fileparts(spk_path);
    if ~strcmpi(spk_ext, '.spk')
        error('extract_spk_with_axisfile_matlab:InvalidExtension', ...
            'Input file must have a .spk extension: %s', spk_path);
    end

    log_progress('Loading trigger-aligned spikes via AxisFile(...).SpikeData.LoadAllSpikes ...');
    axis_file = AxisFile(spk_path);
    spike_data = axis_file.SpikeData;
    if numel(spike_data) ~= 1
        error('extract_spk_with_axisfile_matlab:UnexpectedSpikeDataSets', ...
            'Expected one spike dataset, found %d.', numel(spike_data));
    end

    [hardware_channels, timestamp] = spike_data.LoadAllSpikes;
    timestamp = timestamp(:);
    total_rows = numel(timestamp);
    log_progress(sprintf('Loaded %d trigger-aligned spikes.', total_rows));

    if total_rows == 0
        final_results = create_empty_result_table();
    else
        if numel(hardware_channels.Achk) ~= total_rows || ...
                numel(hardware_channels.Channel) ~= total_rows
            error('extract_spk_with_axisfile_matlab:SpikeDataLengthMismatch', ...
                'LoadAllSpikes returned inconsistent channel and timestamp lengths.');
        end

        hardware_keys = bitor( ...
            bitshift(uint16(hardware_channels.Achk(:)), 8), ...
            uint16(hardware_channels.Channel(:)) ...
        );
        [~, first_channel_indices, channel_group_indices] = unique(hardware_keys);
        unique_channel_count = numel(first_channel_indices);
        log_progress(sprintf( ...
            'Mapping %d unique hardware channels through ChannelArray.', ...
            unique_channel_count));

        channel_mappings = spike_data.ChannelArray.LookupChannelMapping( ...
            hardware_channels.Achk(first_channel_indices), ...
            hardware_channels.Channel(first_channel_indices) ...
        );
        mapped_well_rows = double([channel_mappings.WellRow]).';
        mapped_well_columns = double([channel_mappings.WellColumn]).';
        mapped_electrode_columns = double([channel_mappings.ElectrodeColumn]).';
        mapped_electrode_rows = double([channel_mappings.ElectrodeRow]).';

        well_rows = mapped_well_rows(channel_group_indices);
        well_columns = mapped_well_columns(channel_group_indices);
        electrode_columns = mapped_electrode_columns(channel_group_indices);
        electrode_rows = mapped_electrode_rows(channel_group_indices);

        well_label = arrayfun( ...
            @well_label_from_indices, ...
            well_rows, ...
            well_columns, ...
            'UniformOutput', false ...
        );
        channel_label = arrayfun( ...
            @channel_label_from_indices, ...
            electrode_columns, ...
            electrode_rows ...
        );

        final_results = table( ...
            channel_label(:), ...
            well_label(:), ...
            timestamp, ...
            'VariableNames', { ...
                'Channel_Label', ...
                'Well_Label', ...
                'Timestamp' ...
            } ...
        );
        log_progress(sprintf( ...
            'Mapped %d spikes across %d hardware channels.', ...
            total_rows, unique_channel_count));
    end

    log_progress(sprintf('Writing CSV table with %d rows.', height(final_results)));
    writetable(final_results, output_csv_path);
    log_progress(sprintf('Completed extraction in %.1fs.', toc(run_timer)));
    fprintf('Wrote CSV: %s\n', output_csv_path);
end

function log_progress(message)
    fprintf('[%s] %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), message);
    drawnow();
end

function label = well_label_from_indices(well_row, well_column)
    label = sprintf('%s%d', char(double('A') + well_row - 1), well_column);
end

function label = channel_label_from_indices(electrode_column, electrode_row)
    label = str2double(sprintf('%d%d', electrode_column, electrode_row));
end

function empty_table = create_empty_result_table()
    empty_table = table( ...
        zeros(0, 1), ...
        cell(0, 1), ...
        zeros(0, 1), ...
        'VariableNames', { ...
            'Channel_Label', ...
            'Well_Label', ...
            'Timestamp' ...
        } ...
    );
end
