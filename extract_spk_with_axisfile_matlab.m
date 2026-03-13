function output_csv_path = extract_spk_with_axisfile_matlab(spk_path, output_csv, loader_dir)
%EXTRACT_SPK_WITH_AXISFILE_MATLAB Extract spike timings from one Axion .spk file.
%   output_csv_path = extract_spk_with_axisfile_matlab(spk_path, output_csv, loader_dir)
%   loads spike data with the AxionFileLoader MATLAB API,
%   extracts timing and amplitude columns and writes a .csv file.
%
%   - spk_path: path to input .spk file
%   - output_csv: path to output .csv file (optional; defaults to spk basename)
%   - loader_dir: path to AxionFileLoader class files (optional)

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

    addpath(loader_dir);

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

    all_data = AxisFile(spk_path).SpikeData.LoadData;
    [nwr, nwc, nec, ner] = size(all_data);
    result_blocks = cell(0, 1);

    for wr = 1:nwr
        for wc = 1:nwc
            for ec = 1:nec
                for er = 1:ner
                    data = all_data{wr, wc, ec, er};
                    if isempty(data)
                        continue
                    end

                    [t, v] = data.GetTimeVoltageVector;
                    timestamp = t(1, :);
                    timestamp_length = length(timestamp);

                    channel_label = repmat(str2double(strcat(num2str(ec), num2str(er))), 1, timestamp_length);
                    well_label = repmat(str2double(strcat(num2str(wr), num2str(wc))), 1, timestamp_length);
                    min_amplitude = min(v);
                    max_amplitude = max(v);
                    peak_to_peak_amplitude = max_amplitude - min_amplitude;

                    result_blocks{end + 1, 1} = transpose([ ...
                        channel_label; ...
                        well_label; ...
                        timestamp; ...
                        max_amplitude; ...
                        min_amplitude; ...
                        peak_to_peak_amplitude ...
                    ]);
                end
            end
        end
    end

    if isempty(result_blocks)
        final_results = zeros(0, 6);
    else
        final_results = vertcat(result_blocks{:});
    end

    final_results = array2table(final_results, ...
        'VariableNames', { ...
            'Channel_Label', ...
            'Well_Label', ...
            'Timestamp', ...
            'Maximum_Amplitude', ...
            'Minimum_Amplitude', ...
            'Peak_to_peak_Amplitude' ...
        });

    writetable(final_results, output_csv_path);
    fprintf('Wrote CSV: %s\n', output_csv_path);
end
