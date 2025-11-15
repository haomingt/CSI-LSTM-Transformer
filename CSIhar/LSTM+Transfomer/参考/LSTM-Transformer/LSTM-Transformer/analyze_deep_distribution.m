function dist_model = analyze_deep_distribution(rssi_data, people_count)
    % 深度分布分析与建模
    
    dist_model = struct();
    dist_model.people_count = people_count;
    dist_model.length = length(rssi_data);
    
    % 基础统计特征
    dist_model.mean = mean(rssi_data);
    dist_model.std = std(rssi_data);
    dist_model.skewness = skewness(rssi_data);
    dist_model.kurtosis = kurtosis(rssi_data);
    dist_model.median = median(rssi_data);
    dist_model.iqr = iqr(rssi_data);
    
    % 动态特征分析
    if length(rssi_data) > 2
        diff1 = diff(rssi_data);
        diff2 = diff(diff1);
        dist_model.velocity_mean = mean(abs(diff1));
        dist_model.velocity_std = std(diff1);
        dist_model.acceleration_mean = mean(abs(diff2));
        dist_model.acceleration_std = std(diff2);
        dist_model.complexity_score = sum(abs(diff2)) / (length(rssi_data) * std(rssi_data) + eps);
    else
        dist_model.velocity_mean = 0;
        dist_model.velocity_std = 0;
        dist_model.acceleration_mean = 0;
        dist_model.acceleration_std = 0;
        dist_model.complexity_score = 0;
    end
    
    % 频域特征分析
    if length(rssi_data) >= 128
        fft_data = abs(fft(rssi_data));
        freqs = (0:length(fft_data)-1) / length(fft_data);
        
        dist_model.spectral_centroid = sum(freqs' .* fft_data) / sum(fft_data);
        dist_model.spectral_bandwidth = sqrt(sum(((freqs' - dist_model.spectral_centroid).^2) .* fft_data) / sum(fft_data));
        dist_model.spectral_energy = sum(fft_data.^2);
        
        % 频域分段能量
        n_bands = 8;
        band_size = floor(length(fft_data) / n_bands);
        dist_model.band_energies = zeros(n_bands, 1);
        for b = 1:n_bands
            start_idx = (b-1)*band_size + 1;
            end_idx = min(b*band_size, length(fft_data));
            dist_model.band_energies(b) = sum(fft_data(start_idx:end_idx).^2);
        end
        dist_model.band_energies = dist_model.band_energies / sum(dist_model.band_energies);
    else
        dist_model.spectral_centroid = 0.5;
        dist_model.spectral_bandwidth = 0.1;
        dist_model.spectral_energy = 1;
        dist_model.band_energies = ones(8, 1) / 8;
    end
    
    % 多尺度局部特征
    window_sizes = [10, 25, 50];
    dist_model.local_features = struct();
    
    for w = 1:length(window_sizes)
        ws = window_sizes(w);
        if ws < length(rssi_data)
            local_means = [];
            local_stds = [];
            
            for start = 1:ws:(length(rssi_data)-ws+1)
                window_end = min(start + ws - 1, length(rssi_data));
                window_data = rssi_data(start:window_end);
                local_means(end+1) = mean(window_data);
                local_stds(end+1) = std(window_data);
            end
            
            field_name = sprintf('scale_%d', ws);
            dist_model.local_features.(field_name).mean_var = var(local_means);
            dist_model.local_features.(field_name).std_mean = mean(local_stds);
            dist_model.local_features.(field_name).pattern_consistency = 1 / (1 + var(local_means));
        end
    end
    
    % 人数相关的特征权重
    dist_model.people_weight = people_count / 15;  % 归一化人数权重
    dist_model.signal_strength_factor = 1 + people_count * 0.1;  % 人数影响因子
    
    fprintf('  人数%d分布分析: 复杂度=%.3f, 频谱重心=%.3f, 信号强度因子=%.2f\n', ...
        people_count, dist_model.complexity_score, dist_model.spectral_centroid, dist_model.signal_strength_factor);
end