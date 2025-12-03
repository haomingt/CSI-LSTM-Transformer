function features = extract_prediction_features(sequences, feature_dims)
% 提取预测特征（与训练时相同的特征提取）
n_sequences = size(sequences, 1);
features = zeros(n_sequences, feature_dims);

for i = 1:n_sequences
    seq = sequences(i, :)';
    
    % 基础统计特征 (16维)
    features(i, 1) = mean(seq);
    features(i, 2) = std(seq);
    features(i, 3) = min(seq);
    features(i, 4) = max(seq);
    features(i, 5) = median(seq);
    features(i, 6) = range(seq);
    features(i, 7) = skewness(seq);
    features(i, 8) = kurtosis(seq);
    features(i, 9) = var(seq);
    features(i, 10) = iqr(seq);
    features(i, 11) = mad(seq);
    features(i, 12) = rms(seq);
    features(i, 13) = std(seq) / (abs(mean(seq)) + eps);
    features(i, 14) = sum(seq > mean(seq)) / length(seq);
    features(i, 15) = (max(seq) - min(seq)) / (std(seq) + eps);
    features(i, 16) = sum(abs(seq - median(seq))) / length(seq);
    
    % 动态特征 (16维)
    if length(seq) > 2
        diff1 = diff(seq);
        diff2 = diff(diff1);
        
        features(i, 17) = mean(abs(diff1));
        features(i, 18) = std(diff1);
        features(i, 19) = max(abs(diff1));
        features(i, 20) = sum(diff1 > 0) / length(diff1);
        
        if ~isempty(diff2)
            features(i, 21) = mean(abs(diff2));
            features(i, 22) = std(diff2);
            features(i, 23) = sum(abs(diff2));
            features(i, 24) = sum(diff2 > 0) / length(diff2);
        else
            features(i, 21:24) = 0;
        end
        
        features(i, 25) = sum(seq.^2);
        features(i, 26) = sum(abs(seq));
        features(i, 27) = sum(abs(diff1)) / (length(seq) - 1);
        features(i, 28) = var(diff1);
        
        zero_crossings = sum(diff(sign(seq - mean(seq))) ~= 0);
        features(i, 29) = zero_crossings / length(seq);
        
        [peaks, ~] = findpeaks(seq);
        [valleys, ~] = findpeaks(-seq);
        features(i, 30) = length(peaks) / length(seq);
        features(i, 31) = length(valleys) / length(seq);
        features(i, 32) = (length(peaks) + length(valleys)) / length(seq);
    else
        features(i, 17:32) = 0;
    end
    
    % 频域特征 (16维)
    if length(seq) >= 64
        fft_seq = abs(fft(seq));
        freqs = (0:length(fft_seq)-1) / length(fft_seq);
        
        features(i, 33) = mean(fft_seq);
        features(i, 34) = std(fft_seq);
        features(i, 35) = max(fft_seq);
        features(i, 36) = sum(fft_seq(1:min(10, end)));
        features(i, 37) = sum(fft_seq(max(1, end-9):end));
        features(i, 38) = sum(freqs' .* fft_seq) / sum(fft_seq);
        features(i, 39) = sqrt(sum(((freqs' - features(i, 38)).^2) .* fft_seq) / sum(fft_seq));
        
        cumulative_energy = cumsum(fft_seq);
        rolloff_idx = find(cumulative_energy >= 0.85 * sum(fft_seq), 1);
        if ~isempty(rolloff_idx)
            features(i, 40) = freqs(rolloff_idx);
        else
            features(i, 40) = 0.5;
        end
        
        features(i, 41) = geomean(fft_seq + eps) / (mean(fft_seq) + eps);
        
        [freq_peaks, ~] = findpeaks(fft_seq);
        features(i, 42) = length(freq_peaks) / length(fft_seq);
        
        [~, max_freq_idx] = max(fft_seq);
        features(i, 43) = freqs(max_freq_idx);
        
        features(i, 44) = sum(abs(diff(fft_seq))) / sum(fft_seq);
        
        if length(fft_seq) > 10
            p = polyfit(freqs(1:floor(length(freqs)/2))', fft_seq(1:floor(length(fft_seq)/2)), 1);
            features(i, 45) = p(1);
        else
            features(i, 45) = 0;
        end
        
        n_segments = 4;
        seg_length = floor(length(fft_seq) / n_segments);
        centroids = zeros(n_segments, 1);
        for seg = 1:n_segments
            start_idx = (seg-1)*seg_length + 1;
            end_idx = min(seg*seg_length, length(fft_seq));
            seg_freqs = freqs(start_idx:end_idx);
            seg_fft = fft_seq(start_idx:end_idx);
            centroids(seg) = sum(seg_freqs' .* seg_fft) / sum(seg_fft);
        end
        features(i, 46) = std(centroids);
        features(i, 47) = max(centroids) - min(centroids);
        features(i, 48) = mean(diff(centroids));
    else
        features(i, 33:48) = zeros(1, 16);
    end
    
    % 高级特征 (16维)
    quarter_point = floor(length(seq)/4);
    half_point = floor(length(seq)/2);
    three_quarter_point = floor(3*length(seq)/4);
    
    if quarter_point > 0
        q1 = seq(1:quarter_point);
        q2 = seq(quarter_point+1:half_point);
        q3 = seq(half_point+1:three_quarter_point);
        q4 = seq(three_quarter_point+1:end);
        
        features(i, 49) = mean(q1) - mean(q4);
        features(i, 50) = std(q1) - std(q4);
        features(i, 51) = mean(q2) - mean(q3);
        features(i, 52) = var([mean(q1), mean(q2), mean(q3), mean(q4)]);
    else
        features(i, 49:52) = 0;
    end
    
    if length(seq) > 1
        features(i, 53) = corr(seq(1:end-1), seq(2:end));
        if length(seq) > 5
            features(i, 54) = corr(seq(1:end-5), seq(6:end));
        else
            features(i, 54) = 0;
        end
    else
        features(i, 53:54) = 0;
    end
    
    window_sizes = [10, 25, 50];
    for w_idx = 1:length(window_sizes)
        ws = window_sizes(w_idx);
        if ws < length(seq)
            local_means = movmean(seq, ws);
            features(i, 54 + w_idx) = var(local_means);
        else
            features(i, 54 + w_idx) = 0;
        end
    end
    
    features(i, 58) = sum(abs(seq).^1.5);
    features(i, 59) = mean(exp(-abs(seq - mean(seq))));
    features(i, 60) = sum(seq.^3) / (length(seq) * std(seq)^3 + eps);
    
    [counts, ~] = histcounts(seq, 20);
    probs = counts / sum(counts);
    probs = probs(probs > 0);
    if ~isempty(probs)
        features(i, 61) = -sum(probs .* log2(probs));
    else
        features(i, 61) = 0;
    end
    
    if length(seq) > 2 && var(seq) > eps
        diff2 = diff(diff(seq));
        features(i, 62) = sum(abs(diff2)) / (var(seq) + eps);
    else
        features(i, 62) = 0;
    end
    
    % 简化的分形维数和Hurst指数
    try
        if length(seq) >= 20
            complexity_measure = std(diff(seq)) / (mean(abs(seq)) + eps);
            features(i, 63) = 1 + min(1, complexity_measure);
        else
            features(i, 63) = 1.5;
        end
    catch
        features(i, 63) = 1.5;
    end
    
    try
        if length(seq) >= 10
            lag1_corr = corr(seq(1:end-1), seq(2:end));
            features(i, 64) = 0.5 + 0.3 * lag1_corr;
        else
            features(i, 64) = 0.5;
        end
    catch
        features(i, 64) = 0.5;
    end
end

% 处理NaN和Inf值
features(isnan(features)) = 0;
features(isinf(features)) = 0;

% 特征标准化（简化版）
feature_means = mean(features);
feature_stds = std(features);
features = (features - feature_means) ./ (feature_stds + eps);

% 异常值裁剪
features = max(features, -5);
features = min(features, 5);
end