function aug_seq = band_energy_preserving_transform(original_seq, target_band_energies)
    % 频段能量保持的变换
    if length(original_seq) >= 64
        fft_seq = fft(original_seq);
        
        % 计算当前频段能量
        n_bands = length(target_band_energies);
        band_size = floor(length(fft_seq) / n_bands);
        current_energies = zeros(n_bands, 1);
        
        for b = 1:n_bands
            start_idx = (b-1)*band_size + 1;
            end_idx = min(b*band_size, length(fft_seq));
            current_energies(b) = sum(abs(fft_seq(start_idx:end_idx)).^2);
        end
        
        if sum(current_energies) > 0
            current_energies = current_energies / sum(current_energies);
            
            % 调整频段能量
            for b = 1:n_bands
                start_idx = (b-1)*band_size + 1;
                end_idx = min(b*band_size, length(fft_seq));
                
                if current_energies(b) > 0
                    energy_ratio = sqrt(target_band_energies(b) / current_energies(b));
                    fft_seq(start_idx:end_idx) = fft_seq(start_idx:end_idx) * energy_ratio;
                end
            end
        end
        
        aug_seq = real(ifft(fft_seq));
    else
        aug_seq = original_seq;
    end
end