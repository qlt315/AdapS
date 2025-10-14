clear; clc
markers = {'o','d','s','^','v','>','<','p','h','x','+','*'};
%% Fig-1(a)  –  Top-1 accuracy versus SNR (MATLAB default colours)

data = load('results/fig1a.mat');
acc_matrix_1a = data.acc_matrix_1a;              % M × L
log_names  = string(data.log_names(:));    % M × 1

[num_logs, ~] = size(acc_matrix_1a);


figure; hold on
h = gobjects(num_logs,1);                  % handles for legend

for i = 1:num_logs
    y = acc_matrix_1a(i,:);
    valid = ~isnan(y);
    x = find(valid);                       % assume SNR grid is 1…max_steps
    y = y(valid);

    m_idx = mod(i-1, numel(markers)) + 1;  % cyclic marker index

    % --- draw ---
    h(i) = plot(x, y, ...
        'LineWidth', 4, ...
        'Marker', markers{m_idx}, ...
        'MarkerSize', 8, ...
        'MarkerIndices', 1:3:numel(x));    % sparse markers
end


xlabel('SNR [dB]',          'FontSize',14, 'FontName','Times New Roman')
ylabel('Top-1 Accuracy [%]', 'FontSize',14, 'FontName','Times New Roman')

xlim([1 25])
xticks([1 5 10 15 20 25])
grid on


legend(h, log_names, ...
    'Location','best', ...
    'Interpreter','none', ...
    'FontSize',14, ...
    'FontName','Times New Roman')

ax = gca;
ax.FontSize  = 14;
ax.Box       = 'on';
ax.GridAlpha = 0.5;

hold off




%% Fig‑1(b): Top‑1 Accuracy of different JSCC variants vs. SNR
S = load('results/fig1b.mat');
acc_matrix_1b = S.acc_matrix_1b;
log_names  = string(S.log_names(:));

[num_logs, max_steps] = size(acc_matrix_1b);


sel = 1:num_logs;                     % indices to plot
figure; hold on
h = gobjects(numel(sel),1);           % hold line handles for legend

for k = 1:numel(sel)
    idx  = sel(k);
    y    = acc_matrix_1b(idx,:);
    mask = ~isnan(y);
    x    = find(mask);
    y    = y(mask);

    mk   = markers{ mod(k-1, numel(markers))+1 };  % cyclic marker

    h(k) = plot(x, y, ...
        'LineWidth',      4, ...
        'Marker',         mk, ...
        'MarkerSize',     8, ...
        'MarkerIndices',  1:3:numel(x));   % draw marker every 3 points
end


xlabel('SNR [dB]',          'FontSize',14,'FontName','Times New Roman')
ylabel('Top-1 Accuracy [%]', 'FontSize',14,'FontName','Times New Roman')

xlim([1 25])
xticks([1 5 10 15 20 25])
grid on

legend(h, log_names(sel), ...
    'Interpreter','none', ...
    'Location','best', ...
    'FontSize',14, ...
    'FontName','Times New Roman')

ax = gca;
ax.FontSize  = 14;
ax.Box       = 'on';
ax.GridAlpha = 0.5;

hold off


%% Fig-1(c): Contineous Adaption under different domain shifts
% AWGN -> Rician -> Rayleigh -> AWGN w/ N -> Rician w/ N -> Rayleigh w/ N
domain_labels = {'D1','D2','D3','D4','D5','D6'};
x = 1:numel(domain_labels);


jscc_acc      = [80.58, 65.65, 54.03, 18.87, 16.55, 15.14];
rt_jscc_acc   = [80.58, 75.43, 64.56, 59.95, 62.40, 53.48];
meta_jscc_acc = [76.89, 64.73, 56.29, 42.85, 45.91, 40.08];
jt_jscc_acc   = [82.70, 71.42, 61.39, 53.50, 46.42, 39.89];
md_jscc_acc   = [73.33, 69.73, 41.99, 51.62, 38.60, 31.66];

acc_mat = cat(1, jscc_acc,  rt_jscc_acc, meta_jscc_acc, jt_jscc_acc, md_jscc_acc);
method_names = ["JSCC","RT-JSCC","Meta-JSCC","JT-JSCC","MD-JSCC"];




figure; hold on;

% get default colour-order from current axes
ax  = gca;
cols = ax.ColorOrder;
nCol = size(cols,1);

h = gobjects(numel(method_names),1);

for i = 1:numel(method_names)
    colour = cols(mod(i-1,nCol)+1,:);        % cycle through default colours
    y      = acc_mat(i,:);
    
    h(i) = plot(x, y, ...
        'LineWidth', 4, ...
        'LineStyle','-', ...
        'Color', colour, ...
        'Marker', markers{i}, ...
        'MarkerSize', 8, ...
        'MarkerEdgeColor', colour, ...
        'MarkerFaceColor', colour, ...
        'MarkerIndices', 1:numel(x));        % marker on every point
end


xlim([1 numel(x)]);
ylim([5 90]);

ax.FontSize = 14;
ax.FontName = 'Times New Roman';
ax.Box      = 'on';
grid on;


set(ax,'XTick',x,'XTickLabel',[]);
ylims = ylim;
offset = 2;                                   % vertical offset for text
for i = 1:numel(x)
    text(x(i), ylims(1)-offset, domain_labels{i}, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','top', ...
        'FontSize',14, 'FontName','Times New Roman');
end

ylabel('Top-1 Accuracy [%]','FontSize',14,'FontName','Times New Roman');


legend(h, method_names, ...
       'Location','southwest', ...
       'Interpreter','none', ...
       'FontSize',14, ...
       'FontName','Times New Roman');

ax = gca;
ax.FontSize  = 14;
ax.Box       = 'on';
ax.GridAlpha = 0.5;

hold off;



%% Fig-1(d): Adaptation Time & Energy on Edge Devices  (vertical layout, horizontal bars)
rt_time   = [358.0,  68.1,  33.4,   9.92] * 60;      % RT-JSCC
meta_time = [0.167, 0.0318, 0.0156, 0.0046] * 60;    % Meta-JSCC

rt_energy   = [11.9, 11.4, 22.8, 7.16];              % RT-JSCC (Wh)
meta_energy = [0.0056, 0.0053, 0.0106, 0.00334];     % Meta-JSCC (Wh)

devices = { ...
   'Nano', ...
   'Xavier', ...
   'PC', ...
   'V-100'};

Y = 1:numel(devices);      % y-positions (top → bottom)
bar_h = 0.35;              % bar height for grouped bars
offset = bar_h/2;


figure('Position',[100 100 820 560]);                % taller canvas
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');


nexttile; hold on;
b1 = barh(Y - offset, rt_time,   bar_h);
b2 = barh(Y + offset, meta_time, bar_h);

set(gca, 'YDir','reverse', ...                        
         'YTick',Y, 'YTickLabel',devices, ...
         'FontSize',14, 'FontName','Times New Roman');
xlabel('Adaptation Time [Sec]','FontSize',14,'FontName','Times New Roman');
xlim([0 15]);                                     % adjust as needed
grid on; box on;

legend([b1 b2],{'RT-JSCC','Meta-JSCC'}, ...
       'Location','northeast','FontSize',14);


nexttile; hold on;
b3 = barh(Y - offset, rt_energy,   bar_h);
b4 = barh(Y + offset, meta_energy, bar_h);

set(gca, 'YDir','reverse', ...
         'YTick',Y, 'YTickLabel',devices, ...
         'FontSize',14, 'FontName','Times New Roman');
xlabel('Energy Consumption [Wh]','FontSize',14,'FontName','Times New Roman');
xlim([0 0.01]);                                     
grid on; box on;

legend([b3 b4],{'RT-JSCC','Meta-JSCC'}, ...
       'Location','northeast','FontSize',14);



%% Fig. 2 (a-f)  Acc performance of TTA under different scenarios
% Directory that stores the *.mat files generated in Python
resultsDir  = 'results';

% File suffixes:  fig2a.mat  …  fig2e.mat
figTags     = {'2a','2b','2c','2d','2e'};
filePrefix  = 'fig';                 % ==> fig2a.mat, fig2b.mat, ...

% Pre-create a colour map large enough for the longest legend
% (MATLAB “lines” gives good contrast)
maxRows = 0;
for f = 1:numel(figTags)
    tmp = load(fullfile(resultsDir, sprintf('%s%s.mat',filePrefix,figTags{f})));
    maxRows = max(maxRows, size(tmp.acc_matrix,1));
end
cmap = lines(maxRows);

%% Loop over 5 sub-figures
for f = 1:numel(figTags)

    matPath = fullfile(resultsDir, sprintf('%s%s.mat',filePrefix,figTags{f}));
    if ~isfile(matPath)
        warning('File not found: %s — skipped.', matPath);
        continue
    end

    S          = load(matPath);
    accMatrix  = S.acc_matrix;              % rows = different methods
    logNames   = string(S.log_names(:));

    [numLogs, ~] = size(accMatrix);

    % ---------- new figure ----------
    figure('Name', sprintf('Fig-2(%s)',figTags{f}), ...
           'Position', [100 100 680 480]); hold on;
    h = gobjects(numLogs,1);                % store line handles for legend

    % ---------- plot each row ----------
    for k = 1:numLogs
        y = accMatrix(k,:);
        mask = ~isnan(y);                   % ignore padded NaN
        x = find(mask);  y = y(mask);

        mk   = markers{ mod(k-1,numel(markers) ) + 1 };
        col  = cmap(k,:);

        h(k) = plot(x, y, ...
            'LineWidth',      4, ...
            'Color',          col, ...
            'Marker',         mk, ...
            'MarkerSize',     8, ...
            'MarkerEdgeColor',col, ...
            'MarkerFaceColor',col, ...
            'MarkerIndices',  1:3:numel(x));   % show a marker every 3 points
    end

    % ---------- axes & labels ----------
    xlabel('SNR [dB]',          'FontSize',14,'FontName','Times New Roman');
    ylabel('Top-1 Accuracy [%]','FontSize',14,'FontName','Times New Roman');

    xlim([1 25]);  xticks([1 5 10 15 20 25]);
    grid on;

    % ---------- legend ----------
    legend(h, logNames, ...
        'Interpreter','none', ...
        'Location','best', ...
        'FontSize',14, ...
        'FontName','Times New Roman');

    % ---------- aesthetics ----------
    ax            = gca;
    ax.FontSize   = 14;
    ax.Box        = 'on';
    ax.GridAlpha  = 0.5;

    hold off;
end



%% Fig. 3 Stream Acc performance of TTA under different scenarios
% AWGN -> Rician -> Rayleigh -> AWGN w/ N -> Rician w/ N -> Rayleigh w/ N
domain_labels = {'D1','D2','D3','D4','D5','D6'};
x = 1:numel(domain_labels);


jscc_acc      = [80.58, 65.65, 54.03, 18.87, 16.55, 15.14];
meta_jscc_acc = [76.89, 64.73, 56.29, 42.85, 45.91, 40.08];
rt_jscc_acc   = [80.58, 75.43, 64.56, 59.95, 62.40, 53.48];
jt_jscc_acc   = [82.70, 71.42, 61.39, 53.50, 46.42, 39.89];
md_jscc_acc   = [73.33, 69.73, 41.99, 51.62, 38.60, 31.66];
ttn_jscc_acc  = [80.58, 67.68, 58.43, 68.95, 59.11, 51.45];
acc_mat = cat(1, jscc_acc, meta_jscc_acc, rt_jscc_acc, jt_jscc_acc, md_jscc_acc, ttn_jscc_acc);
method_names = ["JSCC","Meta-JSCC","RT-JSCC","JT-JSCC","MD-JSCC", "TTA-JSCC"];




figure; hold on;

% get default colour-order from current axes
ax  = gca;
cols = ax.ColorOrder;
nCol = size(cols,1);

h = gobjects(numel(method_names),1);

for i = 1:numel(method_names)
    colour = cols(mod(i-1,nCol)+1,:);        % cycle through default colours
    y      = acc_mat(i,:);
    
    h(i) = plot(x, y, ...
        'LineWidth', 4, ...
        'LineStyle','-', ...
        'Color', colour, ...
        'Marker', markers{i}, ...
        'MarkerSize', 8, ...
        'MarkerEdgeColor', colour, ...
        'MarkerFaceColor', colour, ...
        'MarkerIndices', 1:numel(x));        % marker on every point
end


xlim([1 numel(x)]);
ylim([5 90]);

ax.FontSize = 14;
ax.FontName = 'Times New Roman';
ax.Box      = 'on';
grid on;


set(ax,'XTick',x,'XTickLabel',[]);
ylims = ylim;
offset = 2;                                   % vertical offset for text
for i = 1:numel(x)
    text(x(i), ylims(1)-offset, domain_labels{i}, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','top', ...
        'FontSize',14, 'FontName','Times New Roman');
end

ylabel('Top-1 Accuracy [%]','FontSize',14,'FontName','Times New Roman');


legend(h, method_names, ...
       'Location','southwest', ...
       'Interpreter','none', ...
       'FontSize',14, ...
       'FontName','Times New Roman');

ax = gca;
ax.FontSize  = 14;
ax.Box       = 'on';
ax.GridAlpha = 0.5;

hold off;



%% Fig. 4(a)-(c) Impact of batch size / update interval / momentum  on the TTA performance 
% List of .mat files (converted from .pt)
mat_files = {
    'results/ttn_sweep_CIFAR10_rician.mat', ...
    'results/ttn_sweep_CIFAR10_rayleigh.mat', ...
    'results/ttn_sweep_CIFAR10_noise_awgn.mat', ...
    'results/ttn_sweep_CIFAR10_noise_rician.mat', ...
    'results/ttn_sweep_CIFAR10_noise_rayleigh.mat'
};

% Manually defined legend names (same order as mat_files)
legend_list = {'Rician', 'Rayleigh', 'AWGN w/ N', 'Rician w/ N' 'Rayleigh w/ N'};

% Sweep categories
sweep_types = {'sweep_batch_size', 'sweep_interval', 'sweep_m'};
xlabels = {'Batch Size', 'Update Batch Interval', 'Update Momentum'};

% Optional: custom x-axis ticks (will override .mat keys)
xvals_batch_size = [1, 2, 4, 8, 16, 32, 64, 256];
xvals_interval   = [1, 2, 5, 10, 20, 50, 100];
xvals_m          = [1.0, 5.0, 10.0, 50.0, 100.0];
custom_xvals     = {xvals_batch_size, xvals_interval, xvals_m};

% --------- Main Plot Loop ---------
for i = 1:length(sweep_types)
    sweep = sweep_types{i};
    figure;

    for j = 1:length(mat_files)
        data = load(mat_files{j});
        keys_var = [sweep '_keys'];
        acc1_var = [sweep '_acc1'];

        if ~isfield(data, keys_var) || ~isfield(data, acc1_var)
            warning('Missing fields in %s: %s or %s', mat_files{j}, keys_var, acc1_var);
            continue;
        end

        raw_keys = data.(keys_var);               % char array (NxM)
        acc1 = data.(acc1_var);                   % numeric array
        x_default = str2double(cellstr(raw_keys));
        x_override = custom_xvals{i};

        % Align to custom x-axis if provided
        if length(acc1) == length(x_override)
            x = x_override;
        else
            x = x_default;  % fallback
        end

        [x_sorted, idx] = sort(x);
        acc1_sorted = acc1(idx);

plot(x_sorted, acc1_sorted, ['-' markers{j}], ...
     'LineWidth', 4, ...
     'MarkerSize', 8, ...
     'DisplayName', legend_list{j});
        hold on;
        grid on;
        ax = gca;
        ax.FontSize  = 14;
        ax.Box       = 'on';
        ax.GridAlpha = 0.5;
    end
    
    if i == 1
        xlim([0 256]);  % for batch size
    elseif i == 2
        xlim([0 100]);  % for interval
    elseif i == 3
        xlim([0 100]);  % for momentum
    end

    xlabel(xlabels{i});
    ylabel('Top-1 Accuracy (%)');
    legend('Location', 'best');

    set(gca, 'FontName', 'Times New Roman');  % axis
    set(findall(gcf, 'Type', 'text'), 'FontName', 'Times New Roman');  % text

end


%% Fig. 5 Domain shift detection accuracy under different batch size
mat_files = {
    'results/detection_diff_bs_results_kl_kl.mat',                  'Image=KL, Channel=KL';
    'results/detection_diff_bs_results_kl_wasserstein.mat',         'Image=KL, Channel=Wass.';
    'results/detection_diff_bs_results_wasserstein_kl.mat',         'Image=Wass., Channel=KL';
    'results/detection_diff_bs_results_wasserstein_wasserstein.mat','Image=Wass., Channel=Wass.';
};

% All tested batch sizes
bs_all = [1 2 4 8 16 32 64 128 256 512];

figure('Color','w'); hold on; grid on; box on;

for i = 1:size(mat_files,1)
    S = load(mat_files{i,1});
    bs  = double(S.batch_sizes(:));
    acc = double(S.accuracy(:));
    msk = ~isnan(acc);
    bs  = bs(msk);  acc = acc(msk);

    semilogx(bs, acc, '-', ...
        'LineWidth', 4, ...
        'Marker', markers{i}, ...
        'MarkerSize', 8, ...
        'DisplayName', mat_files{i,2});
end

% Set log scale and ticks
set(gca,'XScale','log');

% Keep all ticks but only label a subset to avoid overlap
xt = bs_all;
xt_labels_show = [1 2 4 8 16 32 64 128 256 512]; % which to label
xt_labels = string(xt);
mask = ~ismember(xt, xt_labels_show);
xt_labels(mask) = ""; % hide some labels

set(gca,'XTick',xt,'XTickLabel',xt_labels);

ylim([0 100]);
xlim([min(bs) max(bs)]);
xlabel('Batch Size');
ylabel('Detection Accuracy (%)');
legend('Location','best','NumColumns',1);
set(gca,'FontName','Times New Roman','FontSize',14);

ax = gca;
ax.FontSize  = 14;
ax.Box       = 'on';
ax.GridAlpha = 0.5;


