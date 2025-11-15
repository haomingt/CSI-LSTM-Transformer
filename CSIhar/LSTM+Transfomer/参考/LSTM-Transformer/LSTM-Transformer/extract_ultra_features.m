function feature_matrix = extract_ultra_features(all_sequences, feature_dims)
    % 提取64维超级特征
    n_sequences = size(all_sequences, 1);
    feature_matrix = zeros(n_sequences, feature_dims);
    
    for i = 1:n_sequences
        seq = all_sequences(i, :)';
        
        % === 基础统计特征 (16维) ===
        feature_matrix(i, 1) = mean(seq);
        feature_matrix(i, 2) = std(seq);
        feature_matrix(i, 3) = min(seq);
        feature_matrix(i, 4) = max(seq);
        feature_matrix(i, 5) = median(seq);
        feature_matrix(i, 6) = range(seq);
        feature_matrix(i, 7) = skewness(seq);
        feature_matrix(i, 8) = kurtosis(seq);
        feature_matrix(i, 9) = var(seq);
        feature_matrix(i, 10) = iqr(seq);
        feature_matrix(i, 11) = mad(seq);  % 平均绝对偏差
        feature_matrix(i, 12) = rms(seq);  % 均方根
        feature_matrix(i, 13) = std(seq) / (abs(mean(seq)) + eps);  % 变异系数
        feature_matrix(i, 14) = sum(seq > mean(seq)) / length(seq);  % 超均值比例
        feature_matrix(i, 15) = (max(seq) - min(seq)) / (std(seq) + eps);  % 归一化范围
        feature_matrix(i, 16) = sum(abs(seq - median(seq))) / length(seq);  % 平均绝对偏离中位数
        
        % === 动态特征 (16维) ===
        if length(seq) > 2
            diff1 = diff(seq);
            diff2 = diff(diff1);
            
            feature_matrix(i, 17) = mean(abs(diff1));
            feature_matrix(i, 18) = std(diff1);
            feature_matrix(i, 19) = max(abs(diff1));
            feature_matrix(i, 20) = sum(diff1 > 0) / length(diff1);  % 上升趋势比例
            
            if ~isempty(diff2)
                feature_matrix(i, 21) = mean(abs(diff2));
                feature_matrix(i, 22) = std(diff2);
                feature_matrix(i, 23) = sum(abs(diff2));
                feature_matrix(i, 24) = sum(diff2 > 0) / length(diff2);  % 加速度上升比例
            else
                feature_matrix(i, 21:24) = 0;
            end
            
            feature_matrix(i, 25) = sum(seq.^2);  % 能量
            feature_matrix(i, 26) = sum(abs(seq));  % 总变化
            feature_matrix(i, 27) = sum(abs(diff1)) / (length(seq) - 1);  % 平均变化率
            feature_matrix(i, 28) = var(diff1);  % 变化率方差
            
            % 零交叉率
            zero_crossings = sum(diff(sign(seq - mean(seq))) ~= 0);
            feature_matrix(i, 29) = zero_crossings / length(seq);
            
            % 峰值特征
            [peaks, ~] = findpeaks(seq);
            [valleys, ~] = findpeaks(-seq);
            feature_matrix(i, 30) = length(peaks) / length(seq);  % 峰值密度
            feature_matrix(i, 31) = length(valleys) / length(seq);  % 谷值密度
            feature_matrix(i, 32) = (length(peaks) + length(valleys)) / length(seq);  % 总极值密度
        else
            feature_matrix(i, 17:32) = 0;
        end
        
        % === 频域特征 (16维) ===
        if length(seq) >= 64
            fft_seq = abs(fft(seq));
            freqs = (0:length(fft_seq)-1) / length(fft_seq);
            
            feature_matrix(i, 33) = mean(fft_seq);
            feature_matrix(i, 34) = std(fft_seq);
            feature_matrix(i, 35) = max(fft_seq);
            feature_matrix(i, 36) = sum(fft_seq(1:min(10, end)));  % 低频能量
            feature_matrix(i, 37) = sum(fft_seq(max(1, end-9):end));  % 高频能量
            feature_matrix(i, 38) = sum(freqs' .* fft_seq) / sum(fft_seq);  % 频谱重心
            feature_matrix(i, 39) = sqrt(sum(((freqs' - feature_matrix(i, 38)).^2) .* fft_seq) / sum(fft_seq));  % 频谱带宽
            
            % 频谱滚降
            cumulative_energy = cumsum(fft_seq);
            rolloff_idx = find(cumulative_energy >= 0.85 * sum(fft_seq), 1);
            if ~isempty(rolloff_idx)
                feature_matrix(i, 40) = freqs(rolloff_idx);
            else
                feature_matrix(i, 40) = 0.5;
            end
            
            % 频谱平坦度
            feature_matrix(i, 41) = geomean(fft_seq + eps) / (mean(fft_seq) + eps);
            
            % 频域峰值特征
            [freq_peaks, ~] = findpeaks(fft_seq);
            feature_matrix(i, 42) = length(freq_peaks) / length(fft_seq);  % 频域峰值密度
            
            % 主要频率成分
            [~, max_freq_idx] = max(fft_seq);
            feature_matrix(i, 43) = freqs(max_freq_idx);  % 主频率
            
            % 频谱不规则性
            feature_matrix(i, 44) = sum(abs(diff(fft_seq))) / sum(fft_seq);
            
            % 频谱斜率
            if length(fft_seq) > 10
                p = polyfit(freqs(1:floor(length(freqs)/2))', fft_seq(1:floor(length(fft_seq)/2)), 1);
                feature_matrix(i, 45) = p(1);
            else
                feature_matrix(i, 45) = 0;
            end
            
            % 频谱质心变化
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
            feature_matrix(i, 46) = std(centroids);  % 质心变化
            feature_matrix(i, 47) = max(centroids) - min(centroids);  % 质心范围
            feature_matrix(i, 48) = mean(diff(centroids));  % 质心趋势
        else
            feature_matrix(i, 33:48) = zeros(1, 16);
        end
        
        % === 高级特征 (16维) ===
        % 多尺度对比特征
        quarter_point = floor(length(seq)/4);
        half_point = floor(length(seq)/2);
        three_quarter_point = floor(3*length(seq)/4);
        
        if quarter_point > 0
            q1 = seq(1:quarter_point);
            q2 = seq(quarter_point+1:half_point);
            q3 = seq(half_point+1:three_quarter_point);
            q4 = seq(three_quarter_point+1:end);
            
            feature_matrix(i, 49) = mean(q1) - mean(q4);  % 首尾对比
            feature_matrix(i, 50) = std(q1) - std(q4);   % 首尾变异对比
            feature_matrix(i, 51) = mean(q2) - mean(q3);  % 中间对比
            feature_matrix(i, 52) = var([mean(q1), mean(q2), mean(q3), mean(q4)]);  % 分段均值变异
        else
            feature_matrix(i, 49:52) = 0;
        end
        
        % 自相关特征
        if length(seq) > 1
            feature_matrix(i, 53) = corr(seq(1:end-1), seq(2:end));  % 滞后1自相关
            if length(seq) > 5
                feature_matrix(i, 54) = corr(seq(1:end-5), seq(6:end));  % 滞后5自相关
            else
                feature_matrix(i, 54) = 0;
            end
        else
            feature_matrix(i, 53:54) = 0;
        end
        
        % 多尺度局部特征
        window_sizes = [10, 25, 50];
        for w_idx = 1:length(window_sizes)
            ws = window_sizes(w_idx);
            if ws < length(seq)
                local_means = movmean(seq, ws);
                feature_matrix(i, 54 + w_idx) = var(local_means);
            else
                feature_matrix(i, 54 + w_idx) = 0;
            end
        end
        
        % 非线性特征
        feature_matrix(i, 58) = sum(abs(seq).^1.5);  % 1.5次幂和
        feature_matrix(i, 59) = mean(exp(-abs(seq - mean(seq))));  % 指数衰减特征
        feature_matrix(i, 60) = sum(seq.^3) / (length(seq) * std(seq)^3 + eps);  % 标准化三阶矩
        
        % 熵特征
        [counts, ~] = histcounts(seq, 20);
        probs = counts / sum(counts);
        probs = probs(probs > 0);
        if ~isempty(probs)
            feature_matrix(i, 61) = -sum(probs .* log2(probs));  % Shannon熵
        else
            feature_matrix(i, 61) = 0;
        end
        
        % 复杂度特征
        if length(seq) > 2 && var(seq) > eps
            diff2 = diff(diff(seq));
            feature_matrix(i, 62) = sum(abs(diff2)) / (var(seq) + eps);  % 归一化复杂度
        else
            feature_matrix(i, 62) = 0;
        end
        
        % % 分形维数估计
        % feature_matrix(i, 63) = estimate_fractal_dimension(seq);
        % 
        % % 长程相关性
        % feature_matrix(i, 64) = estimate_hurst_exponent(seq);
        % 分形维数估计（简化版）
if length(seq) >= 20
    feature_matrix(i, 63) = 1 + 0.5 * std(diff(seq)) / (std(seq) + eps);
else
    feature_matrix(i, 63) = 1.5;
end

% 长程相关性（简化版）
if length(seq) >= 10
    autocorr_1 = corr(seq(1:end-1), seq(2:end));
    feature_matrix(i, 64) = 0.5 + 0.5 * autocorr_1;
else
    feature_matrix(i, 64) = 0.5;
end
    end
    
    % 处理NaN和Inf值
    feature_matrix(isnan(feature_matrix)) = 0;
    feature_matrix(isinf(feature_matrix)) = 0;
end
