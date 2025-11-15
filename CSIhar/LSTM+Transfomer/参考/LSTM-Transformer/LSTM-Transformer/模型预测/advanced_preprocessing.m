function processed_data = advanced_preprocessing(rssi_data)
% 高级预处理（与训练时相同）
rssi_data = double(rssi_data(:));

% 多级异常值检测与修复
for iter = 1:5
    Q1 = quantile(rssi_data, 0.05);
    Q3 = quantile(rssi_data, 0.95);
    IQR = Q3 - Q1;
    outliers = rssi_data < (Q1 - 3*IQR) | rssi_data > (Q3 + 3*IQR);
    rssi_data(outliers) = median(rssi_data);
end

% 多尺度自适应滤波
if length(rssi_data) > 100
    window_sizes = [5, 11, 21, 31];
    weights = [0.4, 0.3, 0.2, 0.1];
    filtered_versions = zeros(length(rssi_data), length(window_sizes));
    
    for w = 1:length(window_sizes)
        if window_sizes(w) < length(rssi_data)
            filtered_versions(:, w) = smooth(rssi_data, window_sizes(w));
        else
            filtered_versions(:, w) = rssi_data;
        end
    end
    
    rssi_data = sum(filtered_versions .* weights, 2);
end

% 信号增强与标准化
rssi_data = rssi_data - mean(rssi_data);
rssi_data = rssi_data / (std(rssi_data) + eps);

% 去趋势处理
if length(rssi_data) > 50
    t = (1:length(rssi_data))';
    p = polyfit(t, rssi_data, 2);
    trend = polyval(p, t);
    rssi_data = rssi_data - trend;
end

processed_data = rssi_data;
end