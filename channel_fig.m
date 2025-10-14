clear; clc; close all;

num_samples = 1e6;
SNR_dB = 20;  
SNR = 10^(SNR_dB/10);
sigma_n = 1/sqrt(2*SNR); 

K_dB = 3;
K = 10^(K_dB/10);

edges = 0:0.02:3;

% ===== 1. AWGN  =====
tx = ones(num_samples,1);
noise = sigma_n*(randn(num_samples,1) + 1j*randn(num_samples,1));
rx_awgn = tx + noise;
awgn_amp = abs(rx_awgn);
[count_awgn,~] = histcounts(awgn_amp, edges, 'Normalization', 'pdf');

% ===== 2. Rayleigh  =====
h_ray = (randn(num_samples,1) + 1j*randn(num_samples,1))/sqrt(2);
rx_ray = h_ray .* tx + noise;
ray_amp = abs(rx_ray);
[count_ray,~] = histcounts(ray_amp, edges, 'Normalization', 'pdf');

% ===== 3. Rician  =====
h_rice = sqrt(K/(K+1)) + ...
         (randn(num_samples,1) + 1j*randn(num_samples,1))/sqrt(2*(K+1));
rx_rice = h_rice .* tx + noise;
rice_amp = abs(rx_rice);
[count_rice,~] = histcounts(rice_amp, edges, 'Normalization', 'pdf');


centers = (edges(1:end-1) + edges(2:end))/2;
figure;
plot(centers, count_awgn, 'LineWidth', 4); hold on;
plot(centers, count_ray,'LineWidth', 4);
plot(centers, count_rice, 'LineWidth', 4);
grid on;
xlabel('Amplitude'); ylabel('PDF');
legend('AWGN', 'Rayleigh', 'Rician');
title(sprintf('Amplitude PDF (SNR = %d dB)', SNR_dB));


set(gca, 'FontName', 'Times New Roman');  % axis
set(findall(gcf, 'Type', 'text'), 'FontName', 'Times New Roman');  % text