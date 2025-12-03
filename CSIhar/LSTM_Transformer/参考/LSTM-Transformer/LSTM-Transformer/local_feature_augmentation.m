function aug_seq = local_feature_augmentation(original_seq, dist_model)
    % 基于局部特征的增强
    aug_seq = original_seq;
    
    % 获取局部特征信息
    if isfield(dist_model, 'local_features')
        fields = fieldnames(dist_model.local_features);
        for f = 1:length(fields)
            feature = dist_model.local_features.(fields{f});
            
            % 基于局部特征添加变化
            window_size = str2double(regexp(fields{f}, '\d+', 'match'));
            if ~isempty(window_size) && window_size < length(original_seq)
                n_windows = floor(length(original_seq) / window_size);
                for w = 1:n_windows
                    start_idx = (w-1) * window_size + 1;
                    end_idx = min(start_idx + window_size - 1, length(original_seq));
                    
                    local_variation = feature.std_mean * randn * 0.1;
                    aug_seq(start_idx:end_idx) = aug_seq(start_idx:end_idx) + local_variation;
                end
            end
        end
    end
end