function aug_seq = complexity_preserving_warp(original_seq, target_complexity)
    % 保持复杂度的时间扭曲
    sequence_length = length(original_seq);
    
    % 非线性时间扭曲
    warp_strength = 0.15;
    warp_points = sort(rand(5, 1));
    warp_values = 1 + warp_strength * (rand(5, 1) - 0.5);
    
    original_time = linspace(0, 1, sequence_length);
    warped_time = interp1(warp_points, warp_values, original_time, 'pchip', 'extrap');
    warped_time = cumsum(warped_time);
    warped_time = warped_time / warped_time(end);
    
    aug_seq = interp1(original_time, original_seq, warped_time, 'pchip', 'extrap');
    
    % 调整以保持目标复杂度
    current_complexity = sum(abs(diff(diff(aug_seq)))) / (length(aug_seq) * std(aug_seq) + eps);
    if current_complexity > 0
        complexity_ratio = target_complexity / current_complexity;
        aug_seq = aug_seq + (aug_seq - mean(aug_seq)) * (complexity_ratio - 1) * 0.1;
    end
end