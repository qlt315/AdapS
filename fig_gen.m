clear; clc;
%% Fig-1 JSCC PSNR under different SNR
%% Fig-1(a): JSCC PSNR under different SNR (with auto style and custom legend)
clear; clc;

% -------------------------------
% 1) Load data
% -------------------------------
data = load('fig_1_a.mat');
psnr_matrix = data.psnr_matrix_1a;
log_names = string(data.log_names_1a);

[num_logs, max_steps] = size(psnr_matrix);


style_list = [
    struct('name', 'AWGN',     'color', [0.00, 0.45, 0.74], 'marker', 'o', 'linestyle', '-')
    struct('name', 'Rayleigh', 'color', [214, 132, 56]/255, 'marker', 'd', 'linestyle', '-')
    struct('name', 'Rayleigh_w_noise', 'color', [199, 107, 96]/255, 'marker', 'd', 'linestyle', '-')
    struct('name', 'Rician',   'color', [76, 119, 128]/255, 'marker', 'p', 'linestyle', '-')
    struct('name', 'Rician_w_noise', 'color', [115, 165, 162]/255, 'marker', 'p', 'linestyle', '-')
];

% -------------------------------
% 3) Define custom legend labels
% -------------------------------
legend_map = containers.Map( ...
    ["AWGN", "Rayleigh", "Rayleigh_w_noise", "Rician", "Rician_w_noise"], ...
    ["AWGN", "Rayleigh", "Rayleigh w/ Sensor Noise", "Rician", "Rician w/ Sensor Noise"] ...
);

% -------------------------------
% 4) Indices to plot (all)
% -------------------------------
selected_idx = 1:num_logs;

% -------------------------------
% 5) Create figure and plot
% -------------------------------
figure; hold on;
h_lines = gobjects(length(selected_idx), 1);

for j = 1:length(selected_idx)
    i = selected_idx(j);
    
    psnr_values = psnr_matrix(i, :);
    valid_idx = ~isnan(psnr_values);
    
    x_vals = 1:max_steps;
    x_vals = x_vals(valid_idx);
    y_vals = psnr_values(valid_idx);
    
    % === Find style ===
    this_name = strtrim(log_names(i)); % ensure no extra space
    style_idx = find(strcmp({style_list.name}, this_name), 1);
    
    if isempty(style_idx)
        warning("No style defined for log: %s", this_name);
        continue;
    end
    
    this_color = style_list(style_idx).color;
    this_marker = style_list(style_idx).marker;
    this_linestyle = style_list(style_idx).linestyle;
    
    % === Plot with sparse markers every 3 points ===
    h_lines(j) = plot(x_vals, y_vals, ...
        'LineWidth', 3, ...
        'LineStyle', this_linestyle, ...
        'Color', this_color, ...
        'Marker', this_marker, ...
        'MarkerSize', 6, ...
        'MarkerEdgeColor', this_color, ...
        'MarkerFaceColor', this_color, ...
        'MarkerIndices', 1:3:length(x_vals)); % <<< sparse markers
end

% -------------------------------
% 6) Axis labels & ticks
% -------------------------------
xlabel('SNR [dB]', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('PSNR [dB]', 'FontSize', 14, 'FontName', 'Times New Roman');
xlim([1 25]);
xticks([1 5 10 15 20 25]);

% -------------------------------
% 7) Build legend labels
% -------------------------------
legend_labels = strings(1, length(selected_idx));
for j = 1:length(selected_idx)
    this_name = strtrim(log_names(selected_idx(j)));
    if isKey(legend_map, this_name)
        legend_labels(j) = legend_map(this_name);
    else
        legend_labels(j) = this_name; % fallback
    end
end

legend(h_lines, legend_labels, ...
    'Interpreter', 'none', ...
    'Location', 'best', ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman');

% -------------------------------
% 8) Grid and styling
% -------------------------------
grid on;
ax = gca;
ax.FontSize = 14;
ax.GridColor = [0.2 0.2 0.2];
ax.GridAlpha = 0.6;
ax.Box = 'on';

hold off;


%% Fig-2: Different JSCCs PSNR under different SNR
% -------------------------------
% 1) Load data
% -------------------------------
data = load('fig_1_b.mat');
psnr_matrix = data.psnr_matrix_1b;
log_names = string(data.log_names_1b);

[num_logs, max_steps] = size(psnr_matrix);

% -------------------------------
% 2) Define style_list with names
% -------------------------------
style_list = [
    struct('name', 'JSCC',    'color', [0.00, 0.45, 0.74],   'marker', 'o', 'linestyle', '-')
    struct('name', 'JT-JSCC', 'color', [214, 132, 56]/255,   'marker', 'd', 'linestyle', '-')
    struct('name', 'MD-JSCC', 'color', [199, 107, 96]/255,  'marker', 'p', 'linestyle', '-')
    struct('name', 'RT-JSCC', 'color', [76, 119, 128]/255,   'marker', '^', 'linestyle', '-')
];


% -------------------------------
% 3) Define custom legend map
% -------------------------------
legend_map = containers.Map( ...
    ["JSCC", "JT-JSCC", "MD-JSCC", "RT-JSCC"], ...
    ["JSCC", "JT-JSCC", "MD-JSCC", "RT-JSCC"] ...
);

% -------------------------------
% 4) Select logs (by default all)
% -------------------------------
selected_idx = 1:num_logs;

% -------------------------------
% 5) Create figure and plot
% -------------------------------
figure; hold on;
h_lines = gobjects(length(selected_idx), 1);

for j = 1:length(selected_idx)
    i = selected_idx(j);

    psnr_values = psnr_matrix(i, :);
    valid_idx = ~isnan(psnr_values);

    x_vals = 1:max_steps;
    x_vals = x_vals(valid_idx);
    y_vals = psnr_values(valid_idx);

    % === Find style ===
    this_name = strtrim(log_names(i));
    style_idx = find(strcmp({style_list.name}, this_name), 1);

    if isempty(style_idx)
        warning("No style defined for log: %s", this_name);
        continue;
    end

    this_color = style_list(style_idx).color;
    this_marker = style_list(style_idx).marker;
    this_linestyle = style_list(style_idx).linestyle;

    % === Plot with sparse markers every 3 points ===
    h_lines(j) = plot(x_vals, y_vals, ...
        'LineWidth', 3, ...
        'LineStyle', this_linestyle, ...
        'Color', this_color, ...
        'Marker', this_marker, ...
        'MarkerSize', 6, ...
        'MarkerEdgeColor', this_color, ...
        'MarkerFaceColor', this_color, ...
        'MarkerIndices', 1:3:length(x_vals)); % <-- every 3 points
end


xlabel('SNR [dB]', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('PSNR [dB]', 'FontSize', 14, 'FontName', 'Times New Roman');
xlim([1 25]);
xticks([1 5 10 15 20 25]);


legend_labels = strings(1, length(selected_idx));
for j = 1:length(selected_idx)
    this_name = strtrim(log_names(selected_idx(j)));
    if isKey(legend_map, this_name)
        legend_labels(j) = legend_map(this_name);
    else
        legend_labels(j) = this_name;
    end
end

legend(h_lines, legend_labels, ...
    'Interpreter', 'none', ...
    'Location', 'best', ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman');


grid on;
ax = gca;
ax.FontSize = 14;
ax.GridColor = [0.2 0.2 0.2];
ax.GridAlpha = 0.6;
ax.Box = 'on';

hold off;


%% Fig-3: Contineous Adaption under different domain shifts

% -------------------------------
% 1) Single-line X-axis labels
% -------------------------------
domain_labels = {'D1', 'D2', 'D3', 'D4', 'D5'};
x = 1:length(domain_labels);

% -------------------------------
% 2) PSNR values
% -------------------------------
jscc_psnr = [29.66, 10.09, 10.11, 22.6, 18.96];
rt_jscc_psnr = [29.66, 20.88, 17.97, 24.92, 19.72];
jt_jscc_psnr = [21.37, 18.17, 16.36, 20.2, 17.53];
md_jscc_psnr = [24.15, 19.19, 9.04, 21.7, 17.9];

% -------------------------------
% 3) Plot lines
% -------------------------------
figure; hold on;

plot(x, jscc_psnr, '-o', ...
    'LineWidth', 4, 'MarkerSize', 8, ...
    'Color', [0.00, 0.45, 0.74], 'DisplayName', 'JSCC');

plot(x, jt_jscc_psnr, '-d', ...
    'LineWidth', 4, 'MarkerSize', 8, ...
    'Color', [199, 107, 96]/255, 'DisplayName', 'JT-JSCC');

plot(x, md_jscc_psnr, '-p', ...
    'LineWidth', 4, 'MarkerSize', 8, ...
    'Color', [76, 119, 128]/255, 'DisplayName', 'MD-JSCC');

plot(x, rt_jscc_psnr, '-^', ...
    'LineWidth', 4, 'MarkerSize', 8, ...
    'Color', [214, 132, 56]/255, 'DisplayName', 'RT-JSCC');

% -------------------------------
% 4) Adjust axes & ticks
% -------------------------------
xlim([1, 5]);
ylim([8, 30]);
ax = gca;
ax.XTick = x;
ax.YTickMode = 'auto';

% Remove default XTickLabel, we draw custom text
set(gca, 'XTickLabel', []);

% Add custom single-line X labels
yl = ylim;
y_text = yl(1) - 0.5;   % Adjust offset below min y for spacing

for i = 1:5
    text(x(i), y_text, domain_labels{i}, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', ...
        'FontSize', 14, ...
        'FontName', 'Times New Roman', ...
        'Units', 'data');
end

% -------------------------------
% 5) Label & legend
% -------------------------------
ylabel('PSNR [dB]', 'FontSize', 14, 'FontName', 'Times New Roman');

legend('Location', 'southwest', ...
    'FontSize', 14, 'FontName', 'Times New Roman');

% -------------------------------
% 6) Style
% -------------------------------
grid on;
ax.FontSize = 14;
ax.FontName = 'Times New Roman';
ax.GridColor = [0.2 0.2 0.2];
ax.GridAlpha = 0.6;
ax.Box = 'on';

hold off;



%% Fig-4: Energy consumption on different edge computing platforms

% Data: Adaptation times (min) and energy consumption (Wh) for full retraining
origin_times = [7440, 2480, 1488, 372];
origin_energy = [620, 620, 744, 1860];

% Device labels (split into two lines)
device_labels = {
    'Jetson\newline Nano', ...
    'Jetson\newline Xavier', ...
    'Tesla\newline V-100', ...
    'RTX 4080\newline Super PC'
};

% X positions for each device
X = 1:4;

% Small offset to separate the two bars for each device
offset = 0.15;

% Create new figure
figure;

% -------------------------------
% Plot left Y-axis: Adaptation Time
% -------------------------------
yyaxis left
b1 = bar(X - offset, origin_times, 0.3, 'FaceColor', [0.2 0.6 0.8]);
ylabel('Adaptation Time [Min]', ...
    'FontSize', 14, 'FontName', 'Times New Roman');
ylim([0, max(origin_times)*1.1]);

% -------------------------------
% Plot right Y-axis: Energy Consumption
% -------------------------------
yyaxis right
b2 = bar(X + offset, origin_energy, 0.3, 'FaceColor', [0.8 0.4 0.4]);
ylabel('Energy Consumption [Wh]', ...
    'FontSize', 14, 'FontName', 'Times New Roman');
ylim([0, max(origin_energy)*1.1]);

% -------------------------------
% X-axis settings
% -------------------------------
set(gca, 'XTick', X);                     % Position of ticks
set(gca, 'XTickLabel', device_labels);    % Multi-line labels
xlim([0.5, 4.5]);
xtickangle(0);                            % Keep labels horizontal

% -------------------------------
% Add legend
% -------------------------------
legend([b1, b2], {'Adaptation Time', 'Energy Consumption'}, ...
    'FontSize', 14, 'FontName', 'Times New Roman', ...
    'Location', 'northwest');

% -------------------------------
% General axes style
% -------------------------------
ax = gca;
ax.FontSize = 14;
ax.FontName = 'Times New Roman';
ax.GridColor = [0.2 0.2 0.2];
ax.GridAlpha = 0.6;
ax.Box = 'on';
grid on;

