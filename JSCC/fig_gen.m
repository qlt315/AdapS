% Fig-1 JSCC PSNR under different SNR
% Load PSNR data and log names
load('motivation_psnr_data.mat');  % Loads 'psnr_matrix' and 'log_names'

[num_logs, max_steps] = size(psnr_matrix);

% Select only logs that contain "AWGN" or "Rayleigh" in their name
selected_idx = find(contains(log_names, 'AWGN') | contains(log_names, 'Rayleigh') | contains(log_names, 'Rician'));

% Create figure
figure;
hold on;

% Define styles for all logs (color, marker, line style)
% Adjust line styles for better visual distinction
style_list = {
    {[0.00, 0.45, 0.74], 'o', '-'},     % AWGN
    {[214, 132, 56]/255, 'd', '-'},     % Rayleigh
    {[199, 107, 96]/255, 'd', '-'},     % Rayleigh + Noise
    {[76, 119, 128]/255, 'p', '-'},     % Rician
    {[115, 165, 162]/255, 'p', '-'},    % Rician + Noise
};

% Preallocate legend handles
h_lines = gobjects(length(selected_idx), 1);

% Plot selected curves
for j = 1:length(selected_idx)
    i = selected_idx(j);  % Original index from full dataset
    
    psnr_values = psnr_matrix(i, :);
    valid_idx = ~isnan(psnr_values);  % Exclude NaNs
    
    x_vals_all = find(valid_idx);         % Step indices
    y_vals_all = psnr_values(valid_idx);  % PSNR values
    
    % Downsample data (plot every 2nd point)
    x_vals_down = x_vals_all(1:2:end);
    y_vals_down = y_vals_all(1:2:end);
    
    % Get line style
    style = style_list{i};
    this_color = style{1};
    this_marker = style{2};
    this_linestyle = style{3};
    
    % Plot with thick line and solid marker
    h_lines(j) = plot(x_vals_down, y_vals_down, ...
        'LineWidth', 3, ...
        'LineStyle', this_linestyle, ...
        'Color', this_color, ...
        'Marker', this_marker, ...
        'MarkerSize', 6, ...
        'MarkerEdgeColor', this_color, ...
        'MarkerFaceColor', this_color);  % Solid marker
end

% Axis labels and limits
xlabel('SNR [dB]', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('PSNR [dB]', 'FontSize', 14, 'FontName', 'Times New Roman');
xlim([1 25]);
xticks([1, 5, 10, 15, 20, 25]);

% Add legend using selected log names
legend(h_lines, log_names(selected_idx), ...
    'Interpreter', 'none', ...
    'Location', 'best', ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman');

% Grid and plot styling
grid on;
ax = gca;
ax.FontSize = 14;
ax.GridColor = [0.2 0.2 0.2];   % Darker grid color
ax.GridAlpha = 0.6;            % Grid transparency
ax.Box = 'on';                 % Draw box around axes
hold off;



% Fig-2: Different JSCCs PSNR under different SNR

[num_logs, max_steps] = size(psnr_matrix);

% Select only logs that contain "AWGN" or "Rayleigh" in their name
selected_idx = find(contains(log_names, 'AWGN') | contains(log_names, 'Rayleigh') | contains(log_names, 'Rician'));

% Create figure
figure;
hold on;

% Define styles for all logs (color, marker, line style)
% Adjust line styles for better visual distinction
style_list = {
    {[0.00, 0.45, 0.74], 'o', '-'},     % AWGN
    {[214, 132, 56]/255, 'd', '-'},     % Rayleigh
    {[199, 107, 96]/255, 'd', '-'},     % Rayleigh + Noise
    {[76, 119, 128]/255, 'p', '-'},     % Rician
    {[115, 165, 162]/255, 'p', '-'},    % Rician + Noise

};

% Preallocate legend handles
h_lines = gobjects(length(selected_idx), 1);

% Plot selected curves
for j = 1:length(selected_idx)
    i = selected_idx(j);  % Original index from full dataset
    
    psnr_values = psnr_matrix(i, :);
    valid_idx = ~isnan(psnr_values);  % Exclude NaNs
    
    x_vals_all = find(valid_idx);         % Step indices
    y_vals_all = psnr_values(valid_idx);  % PSNR values
    
    % Downsample data (plot every 2nd point)
    x_vals_down = x_vals_all(1:2:end);
    y_vals_down = y_vals_all(1:2:end);
    
    % Get line style
    style = style_list{i};
    this_color = style{1};
    this_marker = style{2};
    this_linestyle = style{3};
    
    % Plot with thick line and solid marker
    h_lines(j) = plot(x_vals_down, y_vals_down, ...
        'LineWidth', 3, ...
        'LineStyle', this_linestyle, ...
        'Color', this_color, ...
        'Marker', this_marker, ...
        'MarkerSize', 6, ...
        'MarkerEdgeColor', this_color, ...
        'MarkerFaceColor', this_color);  % Solid marker
end

% Axis labels and limits
xlabel('SNR [dB]', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('PSNR [dB]', 'FontSize', 14, 'FontName', 'Times New Roman');
xlim([1 25]);
xticks([1, 5, 10, 15, 20, 25]);

% Add legend using selected log names
legend(h_lines, log_names(selected_idx), ...
    'Interpreter', 'none', ...
    'Location', 'best', ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman');

% Grid and plot styling
grid on;
ax = gca;
ax.FontSize = 14;
ax.GridColor = [0.2 0.2 0.2];   % Darker grid color
ax.GridAlpha = 0.6;            % Grid transparency
ax.Box = 'on';                 % Draw box around axes
hold off;






% Fig-1(c): Contineous Adaption under different domain shifts

% Domain labels (for custom 2-line x-tick labels)
domain_labels = {
    {'AWGN', ''}, ...
    {'Rayleigh', ''}, ...
    {'Rayleigh', 'Noise'}, ...
    {'Rician', 'Noise'}, ...
    {'AWGN', ''}
};
x = 1:length(domain_labels);

% PSNR values
jscc_psnr = [25.0, 20.5, 19.0, 22.5, 20.2];
rt_jscc_psnr = [25.0, 23.4, 22.2, 24.0, 23.0];
jt_jscc_psnr = [25.1, 24.0, 23.5, 24.5, 23.9];
md_jscc_psnr = [21.1, 20.0, 19.5, 20.5, 19.9];

% Plot
figure;
hold on;

plot(x, jscc_psnr, '-o', ...
    'LineWidth', 4, ...
    'MarkerSize', 8, ...
    'Color', [0.00, 0.45, 0.74], ...
    'DisplayName', 'JSCC');

plot(x, rt_jscc_psnr, '-s', ...
    'LineWidth', 4, ...
    'MarkerSize', 8, ...
    'Color', [214, 132, 56]/255, ...
    'DisplayName', 'RT-JSCC');

plot(x, jt_jscc_psnr, '-^', ...
    'LineWidth', 4, ...
    'MarkerSize', 8, ...
    'Color', [199, 107, 96]/255, ...
    'DisplayName', 'JT-jSCC');


plot(x, md_jscc_psnr, '-h', ...
    'LineWidth', 4, ...
    'MarkerSize', 8, ...
    'Color', [76, 119, 128]/255, ...
    'DisplayName', 'MD-JSCC');

% Adjust axes
xlim([0.5, 5.5]);
ylim([18, 26]);
set(gca, 'XTick', []);
set(gca, 'XTickLabel', []);
set(gca, 'XTick', 1:5);

% Add custom two-line x-axis labels using `text`
ax = gca;
ax_pos = get(ax, 'Position');
ax_pos(2) = ax_pos(2) + 0.06; % shift plot up
set(ax, 'Position', ax_pos);

for i = 1:5
    label = domain_labels{i};
    text(i, 17.3, sprintf('%s\n%s', label{1}, label{2}), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', ...
        'FontSize', 14, ...
        'FontName', 'Times New Roman', ...
        'Units', 'data');
end

% Labels & legend
ylabel('PSNR [dB]', ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman');

legend('Location', 'southwest', ...
    'FontSize', 14, ...
    'FontName', 'Times New Roman');

% Style
xlim('tight');
ax.FontSize = 14;
ax.FontName = 'Times New Roman';
ax.GridColor = [0.2 0.2 0.2];
ax.GridAlpha = 0.6;
ax.Box = 'on';
grid on;
hold off;



% Fig-4: Energy consumption on different edge computing platforms

% Data: Adaptation times (min) and energy consumption (Wh) for full retraining
origin_times = [7440, 2480, 1488, 372];
origin_energy = [620, 620, 744, 1860];

% Device labels (split into two lines)
device_labels = {
    'Jetson\newline Nano', ...
    'Jetson\newline Xavier NX', ...
    'NVIDIA\newline Orin', ...
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

