function normalized_features = advanced_feature_normalization(feature_matrix)
    % 高级特征标准化
    
    % 第1步：移除常数特征
    feature_std = std(feature_matrix);
    valid_features = feature_std > eps;
    feature_matrix = feature_matrix(:, valid_features);
    
    % 第2步：异常值处理
    for col = 1:size(feature_matrix, 2)
        Q1 = quantile(feature_matrix(:, col), 0.25);
        Q3 = quantile(feature_matrix(:, col), 0.75);
        IQR = Q3 - Q1;
        outliers = feature_matrix(:, col) < (Q1 - 3*IQR) | feature_matrix(:, col) > (Q3 + 3*IQR);
        feature_matrix(outliers, col) = median(feature_matrix(:, col));
    end
    
    % 第3步：Z-score标准化
    feature_means = mean(feature_matrix);
    feature_stds = std(feature_matrix);
    normalized_features = (feature_matrix - feature_means) ./ (feature_stds + eps);
    
    % 第4步：补齐删除的特征（用零填充）
    if size(normalized_features, 2) < 64
        full_features = zeros(size(normalized_features, 1), 64);
        full_features(:, valid_features) = normalized_features;
        normalized_features = full_features;
    end
    
    % 第5步：最终异常值裁剪
    normalized_features = max(normalized_features, -5);
    normalized_features = min(normalized_features, 5);
end
