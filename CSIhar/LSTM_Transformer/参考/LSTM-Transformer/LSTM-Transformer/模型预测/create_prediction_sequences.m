function sequences = create_prediction_sequences(processed_data, sequence_length)
% 创建预测序列
if length(processed_data) >= sequence_length
    % 使用滑动窗口创建多个序列
    overlap_step = max(1, floor(sequence_length / 10));  % 10%重叠
    n_sequences = floor((length(processed_data) - sequence_length) / overlap_step) + 1;
    
    sequences = zeros(n_sequences, sequence_length);
    
    for i = 1:n_sequences
        start_idx = (i-1) * overlap_step + 1;
        end_idx = start_idx + sequence_length - 1;
        if end_idx <= length(processed_data)
            sequences(i, :) = processed_data(start_idx:end_idx)';
        end
    end
    
    % 移除全零行
    sequences = sequences(any(sequences, 2), :);
else
    % 对于短数据，智能扩展
    expansion_factor = ceil(sequence_length / length(processed_data));
    expanded_data = repmat(processed_data, expansion_factor, 1);
    
    % 添加小幅变化使扩展更自然
    for rep = 2:expansion_factor
        start_idx = (rep-1) * length(processed_data) + 1;
        end_idx = rep * length(processed_data);
        variation = 0.1 * std(processed_data) * randn(length(processed_data), 1);
        expanded_data(start_idx:end_idx) = expanded_data(start_idx:end_idx) + variation;
    end
    
    sequences = expanded_data(1:sequence_length)';
    sequences = reshape(sequences, 1, sequence_length);
end
end
