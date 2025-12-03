function [X_train, X_val, X_test, y_train, y_val, y_test] = stratified_split_enhanced(features, all_labels)
    % 增强的分层数据划分
    unique_labels = unique(all_labels);
    X_train = {}; X_val = {}; X_test = {};
    y_train = []; y_val = []; y_test = [];
    
    for label = unique_labels'
        indices = find(all_labels == label);
        n = length(indices);
        
        % 确保每个分组都有足够样本
        if n < 10
            fprintf('警告：人数%d只有%d个样本，可能影响性能\n', label, n);
        end
        
        % 分层随机采样
        indices = indices(randperm(n));
        
        % 优化的划分比例
        if n >= 500
            train_ratio = 0.7; val_ratio = 0.15; test_ratio = 0.15;
        elseif n >= 200
            train_ratio = 0.75; val_ratio = 0.15; test_ratio = 0.1;
        elseif n >= 100
            train_ratio = 0.8; val_ratio = 0.1; test_ratio = 0.1;
        else
            train_ratio = 0.85; val_ratio = 0.1; test_ratio = 0.05;
        end
        
        n_train = max(1, floor(train_ratio * n));
        n_val = max(1, floor(val_ratio * n));
        n_test = max(1, n - n_train - n_val);
        
        train_idx = indices(1:n_train);
        val_idx = indices(n_train+1:n_train+n_val);
        test_idx = indices(n_train+n_val+1:n_train+n_val+n_test);
        
        X_train = [X_train; features(train_idx)];
        X_val = [X_val; features(val_idx)];
        X_test = [X_test; features(test_idx)];
        
        y_train = [y_train; all_labels(train_idx)];
        y_val = [y_val; all_labels(val_idx)];
        y_test = [y_test; all_labels(test_idx)];
    end
end

