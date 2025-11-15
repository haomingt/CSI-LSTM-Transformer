function augmented_sequences = prediction_data_augmentation(base_sequences, augment_needed, sequence_length)
% 预测阶段的数据增强函数
% 输入：
%   base_sequences: 基础序列 [n_seq x seq_length]
%   augment_needed: 需要增强的数量
%   sequence_length: 序列长度
% 输出：
%   augmented_sequences: 增强后的序列

augmented_sequences = zeros(augment_needed, sequence_length);

% 计算基础序列的统计特征
if size(base_sequences, 1) > 0
    all_data = base_sequences(:);
    data_mean = mean(all_data);
    data_std = std(all_data);
    data_range = [min(all_data), max(all_data)];
else
    % 如果没有基础序列，使用默认参数
    data_mean = 0;
    data_std = 1;
    data_range = [-3, 3];
end

for aug_idx = 1:augment_needed
    % 8种保守的增强技术（适合预测阶段）
    technique = mod(aug_idx-1, 8) + 1;
    
    if size(base_sequences, 1) > 0
        % 随机选择一个基础序列
        base_idx = randi(size(base_sequences, 1));
        original_seq = base_sequences(base_idx, :)';
    else
        % 生成基础序列
        original_seq = data_mean + data_std * randn(sequence_length, 1);
    end
    
    switch technique
        case 1  % 轻微时间扭曲
            % 保守的时间扭曲，变形幅度较小
            warp_strength = 0.05;  % 5%的扭曲强度
            warp_points = [0, 0.25, 0.5, 0.75, 1];
            warp_values = 1 + warp_strength * (rand(5, 1) - 0.5);
            
            original_time = linspace(0, 1, sequence_length);
            warped_time = interp1(warp_points, warp_values, original_time, 'linear', 'extrap');
            warped_time = cumsum(warped_time);
            warped_time = warped_time / warped_time(end);
            
            aug_seq = interp1(original_time, original_seq, warped_time, 'linear', 'extrap');
            
        case 2  % 轻微幅度缩放
            % 温和的幅度变换
            scale_factor = 1 + (rand - 0.5) * 0.1;  % ±5%的缩放
            aug_seq = original_seq * scale_factor;
            
        case 3  % 轻微噪声注入
            % 添加少量高斯噪声
            noise_level = data_std * 0.05;  % 5%的噪声水平
            noise = noise_level * randn(sequence_length, 1);
            aug_seq = original_seq + noise;
            
        case 4  % 轻微频域变换
            if sequence_length >= 32
                % 温和的频域修改
                fft_seq = fft(original_seq);
                freq_mask = 1 + (rand(length(fft_seq), 1) - 0.5) * 0.05;  % ±2.5%的频域变化
                modified_fft = fft_seq .* freq_mask;
                aug_seq = real(ifft(modified_fft));
            else
                aug_seq = original_seq + data_std * 0.03 * randn(sequence_length, 1);
            end
            
        case 5  % 局部微调
            % 随机选择小段进行轻微调整
            segment_length = max(5, floor(sequence_length * 0.1));  % 10%的段长度
            segment_start = randi([1, sequence_length - segment_length + 1]);
            segment_end = segment_start + segment_length - 1;
            
            aug_seq = original_seq;
            adjustment = data_std * 0.1 * randn(segment_length, 1);
            aug_seq(segment_start:segment_end) = aug_seq(segment_start:segment_end) + adjustment;
            
        case 6  % 平滑处理
            % 轻微平滑，减少高频噪声
            if sequence_length > 10
                window_size = max(3, floor(sequence_length * 0.05));
                aug_seq = smooth(original_seq, window_size);
            else
                aug_seq = original_seq;
            end
            
        case 7  % 轻微趋势添加
            % 添加轻微的线性或非线性趋势
            if rand > 0.5
                % 线性趋势
                trend_strength = data_std * 0.1;
                trend = linspace(-trend_strength, trend_strength, sequence_length)';
            else
                % 正弦趋势
                trend_strength = data_std * 0.05;
                trend = trend_strength * sin((1:sequence_length)' * 2 * pi / sequence_length * rand);
            end
            aug_seq = original_seq + trend;
            
        case 8  % 序列混合
            if size(base_sequences, 1) >= 2
                % 两个序列的轻微混合
                idx2 = randi(size(base_sequences, 1));
                while idx2 == base_idx && size(base_sequences, 1) > 1
                    idx2 = randi(size(base_sequences, 1));
                end
                seq2 = base_sequences(idx2, :)';
                
                % 轻微混合（90%-10%）
                mix_ratio = 0.05 + 0.1 * rand;  % 5%-15%的混合比例
                aug_seq = (1 - mix_ratio) * original_seq + mix_ratio * seq2;
            else
                % 如果只有一个序列，添加轻微变化
                aug_seq = original_seq + data_std * 0.05 * randn(sequence_length, 1);
            end
    end
    
    % 确保增强后的序列在合理范围内
    aug_seq = max(aug_seq, data_range(1) - data_std);
    aug_seq = min(aug_seq, data_range(2) + data_std);
    
    % 额外的异常值处理
    median_val = median(aug_seq);
    mad_val = mad(aug_seq);
    outliers = abs(aug_seq - median_val) > 3 * mad_val;
    aug_seq(outliers) = median_val;
    
    augmented_sequences(aug_idx, :) = aug_seq';
end

% 最终质量检查
for i = 1:size(augmented_sequences, 1)
    seq = augmented_sequences(i, :);
    
    % 检查是否有异常值
    if any(isnan(seq)) || any(isinf(seq))
        % 如果有异常值，用基础序列替换
        if size(base_sequences, 1) > 0
            replacement_idx = randi(size(base_sequences, 1));
            augmented_sequences(i, :) = base_sequences(replacement_idx, :);
        else
            augmented_sequences(i, :) = data_mean + data_std * randn(1, sequence_length);
        end
    end
end

end