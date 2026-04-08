function output_csv_path = extract_spk_with_axisfile_matlab(spk_path, output_csv, loader_dir)
%EXTRACT_SPK_WITH_AXISFILE_MATLAB Extract spike timings from one Axion .spk file.
%   output_csv_path = extract_spk_with_axisfile_matlab(spk_path, output_csv, loader_dir)
%   loads spike data with the AxionFileLoader MATLAB API,
%   extracts timing and amplitude columns and writes a .csv file.
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

    log_progress('Loading spike data via AxisFile(...).SpikeData.LoadData ...');
    all_data = AxisFile(spk_path).SpikeData.LoadData;
    [nwr, nwc, nec, ner] = size(all_data);
    result_blocks = cell(0, 1);
    total_groups = nwr * nwc * nec * ner;
    scanned_groups = 0;
    nonempty_groups = 0;
    total_rows = 0;

    log_progress(sprintf( ...
        'Loaded spike grid: WellRows=%d, WellColumns=%d, ElectrodeColumns=%d, ElectrodeRows=%d (%d groups).', ...
        nwr, nwc, nec, ner, total_groups));

    for wr = 1:nwr
        for wc = 1:nwc
            log_progress(sprintf( ...
                'Scanning well %s (%d of %d).', ...
                well_label_from_indices(wr, wc), ...
                ((wr - 1) * nwc) + wc, ...
                nwr * nwc));
            for ec = 1:nec
                for er = 1:ner
                    scanned_groups = scanned_groups + 1;
                    data = all_data{wr, wc, ec, er};
                    if isempty(data)
                        if mod(scanned_groups, 250) == 0 || scanned_groups == total_groups
                            log_progress(sprintf( ...
                                'Scanned %d/%d groups; non-empty=%d; rows=%d; elapsed=%.1fs.', ...
                                scanned_groups, total_groups, nonempty_groups, total_rows, toc(run_timer)));
                        end
                        continue
                    end

                    [t, v] = data.GetTimeVoltageVector;
                    timestamp = t(1, :);
                    timestamp_length = length(timestamp);
                    nonempty_groups = nonempty_groups + 1;
                    total_rows = total_rows + timestamp_length;

                    channel_label = repmat(str2double(strcat(num2str(ec), num2str(er))), timestamp_length, 1);
                    well_label = repmat({well_label_from_indices(wr, wc)}, timestamp_length, 1);
                    timestamp = timestamp(:);
                    min_amplitude = min(v, [], 1)';
                    max_amplitude = max(v, [], 1)';
                    peak_to_peak_amplitude = max_amplitude - min_amplitude;

                    result_blocks{end + 1, 1} = table( ...
                        channel_label, ...
                        well_label, ...
                        timestamp, ...
                        max_amplitude, ...
                        min_amplitude, ...
                        peak_to_peak_amplitude, ...
                        'VariableNames', { ...
                            'Channel_Label', ...
                            'Well_Label', ...
                            'Timestamp', ...
                            'Maximum_Amplitude', ...
                            'Minimum_Amplitude', ...
                            'Peak_to_peak_Amplitude' ...
                        } ...
                    );

                    if nonempty_groups <= 5 || mod(nonempty_groups, 25) == 0 || scanned_groups == total_groups
                        log_progress(sprintf( ...
                            'Processed group wr=%d wc=%d ec=%d er=%d; spikes=%d; non-empty=%d; rows=%d; elapsed=%.1fs.', ...
                            wr, wc, ec, er, timestamp_length, nonempty_groups, total_rows, toc(run_timer)));
                    end
                end
            end
        end
    end

    if isempty(result_blocks)
        final_results = create_empty_result_table();
    else
        final_results = vertcat(result_blocks{:});
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

function empty_table = create_empty_result_table()
    empty_table = table( ...
        zeros(0, 1), ...
        cell(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        zeros(0, 1), ...
        'VariableNames', { ...
            'Channel_Label', ...
            'Well_Label', ...
            'Timestamp', ...
            'Maximum_Amplitude', ...
            'Minimum_Amplitude', ...
            'Peak_to_peak_Amplitude' ...
        } ...
    );
end
