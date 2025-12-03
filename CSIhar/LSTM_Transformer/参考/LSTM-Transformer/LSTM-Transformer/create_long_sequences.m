function long_sequences = create_long_sequences(rssi_data, sequence_length, overlap_step)
    % 创建采样点的长序列
    
    if length(rssi_data) >= sequence_length
        n_sequences = floor((length(rssi_data) - sequence_length) / overlap_step) + 1;
        long_sequences = zeros(n_sequences, sequence_length);
        
        for i = 1:n_sequences
            start_idx = (i-1) * overlap_step + 1;
            end_idx = start_idx + sequence_length - 1;
            if end_idx <= length(rssi_data)
                long_sequences(i, :) = rssi_data(start_idx:end_idx)';
            end
        end
        
        % 移除全零行
        long_sequences = long_sequences(any(long_sequences, 2), :);
    else
        % 对于短数据，使用智能扩展
        expansion_factor = ceil(sequence_length / length(rssi_data));
        expanded_data = repmat(rssi_data, expansion_factor, 1);
        
        % 添加渐变变化使扩展更自然
        for rep = 2:expansion_factor
            start_idx = (rep-1) * length(rssi_data) + 1;
            end_idx = rep * length(rssi_data);
            variation = 0.1 * std(rssi_data) * randn(length(rssi_data), 1);
            expanded_data(start_idx:end_idx) = expanded_data(start_idx:end_idx) + variation;
        end
        
        long_sequences = expanded_data(1:sequence_length)';
        long_sequences = reshape(long_sequences, 1, sequence_length);
    end
end