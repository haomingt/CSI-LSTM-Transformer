function augmented_sequences = distribution_based_augmentation(base_sequences, dist_model, augment_needed, sequence_length)
    % 基于真实分布的智能数据增强
    
    augmented_sequences = zeros(augment_needed, sequence_length);
    
    for aug_idx = 1:augment_needed
        % 10种基于分布的增强技术
        technique = mod(aug_idx-1, 10) + 1;
        
        if size(base_sequences, 1) > 0
            base_idx = randi(size(base_sequences, 1));
            original_seq = base_sequences(base_idx, :)';
        else
            % 如果没有基础序列，基于分布模型生成
            original_seq = generate_from_distribution(dist_model, sequence_length);
        end
        
        switch technique
            case 1  % 基于分布参数的生成
                new_seq = dist_model.mean + dist_model.std * randn(sequence_length, 1);
                if abs(dist_model.skewness) > 0.5
                    skew_component = dist_model.skewness * 0.3 * randn(sequence_length, 1).^3;
                    new_seq = new_seq + skew_component;
                end
                aug_seq = new_seq;
                
            case 2  % 保持复杂度的时间扭曲
                target_complexity = dist_model.complexity_score;
                aug_seq = complexity_preserving_warp(original_seq, target_complexity);
                
            case 3  % 频谱保持的幅度变换
                target_spectrum = dist_model.spectral_centroid;
                aug_seq = spectrum_preserving_scale(original_seq, target_spectrum);
                
            case 4  % 基于局部特征的增强
                aug_seq = local_feature_augmentation(original_seq, dist_model);
                
            case 5  % 多尺度混合
                if size(base_sequences, 1) >= 2
                    idx2 = randi(size(base_sequences, 1));
                    seq2 = base_sequences(idx2, :)';
                    
                    % 使用分布权重进行混合
                    weight = 0.3 + 0.4 * dist_model.people_weight;
                    aug_seq = weight * original_seq + (1 - weight) * seq2;
                else
                    aug_seq = original_seq + dist_model.std * 0.1 * randn(sequence_length, 1);
                end
                
            case 6  % 人数相关的模式增强
                people_factor = dist_model.signal_strength_factor;
                pattern_enhancement = sin((1:sequence_length)' * 2 * pi / (50 + dist_model.people_count * 5)) * people_factor * 0.5;
                aug_seq = original_seq + pattern_enhancement;
                
            case 7  % 动态特征保持的噪声注入
                noise_level = dist_model.velocity_std * 0.8;
                dynamic_noise = filter([1], [1, -0.7], randn(sequence_length, 1)) * noise_level;
                aug_seq = original_seq + dynamic_noise;
                
            case 8  % 频段能量保持的变换
                aug_seq = band_energy_preserving_transform(original_seq, dist_model.band_energies);
                
            case 9  % 相关性保持的扰动
                correlation_factor = 0.8;  % 保持80%的相关性
                innovation = sqrt(1 - correlation_factor^2) * dist_model.std * randn(sequence_length, 1);
                aug_seq = correlation_factor * original_seq + innovation;
                
            case 10  % 趋势保持的变化
                if length(original_seq) > 10
                    trend = detrend(original_seq, 'linear');
                    new_trend = polyval([randn*0.001, randn*0.1, dist_model.mean], 1:sequence_length);
                    aug_seq = trend + new_trend';
                else
                    aug_seq = original_seq + dist_model.std * 0.2 * randn(sequence_length, 1);
                end
        end
        
        % 确保增强后的序列在合理范围内
        aug_seq = max(aug_seq, dist_model.mean - 4*dist_model.std);
        aug_seq = min(aug_seq, dist_model.mean + 4*dist_model.std);
        
        augmented_sequences(aug_idx, :) = aug_seq';
    end
end
