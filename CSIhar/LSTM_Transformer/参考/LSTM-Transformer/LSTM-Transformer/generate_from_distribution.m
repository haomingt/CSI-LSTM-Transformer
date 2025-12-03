function seq = generate_from_distribution(dist_model, sequence_length)
    % 基于分布模型生成序列
    seq = dist_model.mean + dist_model.std * randn(sequence_length, 1);
    
    % 添加动态特征
    if sequence_length > 2
        velocity = dist_model.velocity_std * randn(sequence_length-1, 1);
        seq(2:end) = seq(1) + cumsum(velocity);
    end
    
    % 添加人数相关的模式
    pattern = sin((1:sequence_length)' * 2 * pi / (60 + dist_model.people_count * 3)) * dist_model.std * 0.3;
    seq = seq + pattern;
end
