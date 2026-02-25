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

    AllData = AxisFile(spk_path).SpikeData.LoadData;
    final_results = [];
    [nwr, nwc, nec, ner] = size(AllData);

    for wr = 1:nwr
        for wc = 1:nwc
            for ec = 1:nec
                for er = 1:ner
                    data = AllData{wr, wc, ec, er};
                    if ~isempty(data)
                        [t, v] = data.GetTimeVoltageVector;
                        timestamp = t(1, :);
                        timestamp_length = length(timestamp);

                        channel_label = str2double(strcat(num2str(ec), num2str(er)));
                        channel_label = repelem(channel_label, timestamp_length);

                        well_label = str2double(strcat(num2str(wr), num2str(wc)));
                        well_label = repelem(well_label, timestamp_length);

                        min_amplitude = min(v);
                        max_amplitude = max(v);
                        peak_to_peak_amplitude = max_amplitude - min_amplitude;

                        combined_data = transpose([
                            channel_label;
                            well_label;
                            timestamp;
                            max_amplitude;
                            min_amplitude;
                            peak_to_peak_amplitude
                        ]);
                        final_results = [final_results; combined_data];
                    end
                end
            end
        end
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
