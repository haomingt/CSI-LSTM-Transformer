function metrics = calculate_comprehensive_metrics(y_true, y_pred)
    % 计算综合评估指标
    metrics = struct();
    
    % 基础回归指标
    metrics.mse = mean((y_true - y_pred).^2);
    metrics.rmse = sqrt(metrics.mse);
    metrics.mae = mean(abs(y_true - y_pred));
    
    % 准确率指标
    metrics.exact_acc = sum(y_true == y_pred) / length(y_true) * 100;
    metrics.tolerance1_acc = sum(abs(y_true - y_pred) <= 1) / length(y_true) * 100;
    metrics.tolerance2_acc = sum(abs(y_true - y_pred) <= 2) / length(y_true) * 100;
    
    % 相关性指标
    if length(unique(y_true)) > 1 && length(unique(y_pred)) > 1
        metrics.correlation = corr(y_true, y_pred);
    else
        metrics.correlation = 0;
    end
    
    % R²指标
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    if ss_tot > eps
        metrics.r2 = 1 - ss_res / ss_tot;
    else
        metrics.r2 = 0;
    end
    
    % MAPE指标
    metrics.mape = mean(abs((y_true - y_pred) ./ (y_true + eps))) * 100;
    
    % MSE指标
    metrics.mse = mean((y_true - y_pred).^2);

    % MAE指标
    metrics.mae = mean(abs(y_true - y_pred));

    % RMSE指标
    metrics.rmse = sqrt(mean((y_true - y_pred).^2));


    % 分类指标
    unique_labels = unique([y_true; y_pred]);
    n_classes = length(unique_labels);
    
    if n_classes > 1
        precision_scores = zeros(n_classes, 1);
        recall_scores = zeros(n_classes, 1);
        
        for i = 1:n_classes
            class_label = unique_labels(i);
            
            tp = sum(y_true == class_label & y_pred == class_label);
            fp = sum(y_true ~= class_label & y_pred == class_label);
            fn = sum(y_true == class_label & y_pred ~= class_label);
            
            if (tp + fp) > 0
                precision_scores(i) = tp / (tp + fp);
            else
                precision_scores(i) = 0;
            end
            
            if (tp + fn) > 0
                recall_scores(i) = tp / (tp + fn);
            else
                recall_scores(i) = 0;
            end
        end
        
        metrics.avg_precision = mean(precision_scores);
        metrics.avg_recall = mean(recall_scores);
        metrics.f1_score = 2 * metrics.avg_precision * metrics.avg_recall / (metrics.avg_precision + metrics.avg_recall + eps);
    else
        metrics.avg_precision = 0;
        metrics.avg_recall = 0;
        metrics.f1_score = 0;
    end
    
    % 稳定性指标
    errors = y_true - y_pred;
    metrics.error_std = std(errors);
    metrics.max_error = max(abs(errors));
    metrics.error_skewness = skewness(errors);
end