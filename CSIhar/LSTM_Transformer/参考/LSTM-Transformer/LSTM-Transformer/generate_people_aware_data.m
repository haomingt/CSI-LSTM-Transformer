function [sim_data, sim_models] = generate_people_aware_data(people_mapping, sequence_length)
    % 生成基于人数规律的高质量模拟数据
    sim_data = {};
    sim_models = struct();
    
    for i = 1:length(people_mapping)
        people_count = people_mapping(i);
        
        % 基于人数的精确信号模型
        base_rssi = -45 - people_count * 1.8;  % 人数与信号强度的线性关系
        noise_variance = 2 + people_count * 0.3;  % 人数与噪声的关系
        correlation_strength = 0.7 + people_count * 0.02;  % 人数与相关性的关系
        
        % 生成多个长序列
        n_sequences = 15;  % 每个人数生成15个基础序列
        all_people_data = [];
        
        for seq_idx = 1:n_sequences
            % 生成超长序列（确保有足够数据切分）
            data_length = sequence_length * 4 + randi(200);
            t = (1:data_length)';
            
            % 构建复杂的人数相关信号模式
            signal = base_rssi * ones(data_length, 1);
            
            % 主要周期成分（人数相关）
            main_period = 80 + people_count * 5;
            signal = signal + 3 * sin(2*pi*t/main_period);
            
            % 次要周期成分
            sub_period = main_period / 3;
            signal = signal + 1.5 * sin(2*pi*t/sub_period + pi/3);
            
            % 人数特有的复杂模式
            if people_count <= 3
                % 低人数：相对稳定，较少干扰
                interference = 0.5 * sin(2*pi*t/(main_period*2));
            elseif people_count <= 7
                % 中等人数：中等复杂度
                interference = 1.0 * sin(2*pi*t/(main_period*1.5)) + 0.5 * sin(2*pi*t/(main_period*0.8));
            else
                % 高人数：高复杂度，多径效应
                interference = 1.5 * sin(2*pi*t/(main_period*1.2)) + ...
                             0.8 * sin(2*pi*t/(main_period*0.6)) + ...
                             0.4 * sin(2*pi*t/(main_period*0.3));
            end
            
            signal = signal + interference;
            
            % 随机游走成分（人数相关强度）
            random_walk_strength = 0.2 + people_count * 0.05;
            random_walk = cumsum(randn(data_length, 1) * random_walk_strength);
            signal = signal + random_walk;
            
            % 突发事件模拟（人数越多，突发事件越频繁）
            burst_probability = people_count / 20;
            burst_points = find(rand(data_length, 1) < burst_probability);
            for bp = burst_points'
                burst_duration = randi([5, 20]);
                burst_end = min(bp + burst_duration, data_length);
                burst_strength = (1 + people_count * 0.2) * randn;
                signal(bp:burst_end) = signal(bp:burst_end) + burst_strength;
            end
            
            % 相关噪声（AR模型）
            noise = noise_variance * randn(data_length, 1);
            for t_idx = 2:data_length
                noise(t_idx) = correlation_strength * noise(t_idx-1) + ...
                              sqrt(1 - correlation_strength^2) * noise(t_idx);
            end
            signal = signal + noise;
            
            % 物理约束
            signal = max(signal, -95);
            signal = min(signal, -25);
            
            all_people_data = [all_people_data; signal];
        end
        
        % 分析生成数据的分布
        dist_model = analyze_deep_distribution(all_people_data, people_count);
        sim_models.(sprintf('people_%d', people_count)) = dist_model;
        
        sim_data{end+1} = struct('data', all_people_data, 'people', people_count, 'dist_model', dist_model);
        
        fprintf('✓ 生成人数%d模拟数据: 长度=%d, 信号强度=%.2f, 复杂度=%.3f\n', ...
            people_count, length(all_people_data), mean(all_people_data), dist_model.complexity_score);
    end
end