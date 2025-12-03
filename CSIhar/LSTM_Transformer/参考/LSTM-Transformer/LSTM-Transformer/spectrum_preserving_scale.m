function aug_seq = spectrum_preserving_scale(original_seq, target_centroid)
    % 频谱保持的幅度变换
    if length(original_seq) >= 64
        fft_seq = fft(original_seq);
        freqs = (0:length(fft_seq)-1) / length(fft_seq);
        
        current_centroid = sum(freqs' .* abs(fft_seq)) / sum(abs(fft_seq));
        centroid_shift = target_centroid - current_centroid;
        
        % 频域调整
        freq_adjustment = exp(1i * 2 * pi * centroid_shift * (0:length(fft_seq)-1)');
        adjusted_fft = fft_seq .* freq_adjustment;
        
        aug_seq = real(ifft(adjusted_fft));
    else
        aug_seq = original_seq;
    end
    
    % 幅度调整
    scale_factor = 1 + (rand - 0.5) * 0.2;
    aug_seq = aug_seq * scale_factor;
end