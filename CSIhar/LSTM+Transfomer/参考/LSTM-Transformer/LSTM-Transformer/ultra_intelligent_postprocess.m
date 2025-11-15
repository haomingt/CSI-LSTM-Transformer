function y_pred_processed = ultra_intelligent_postprocess(raw_predictions, all_labels, true_labels)
    % 超强智能后处理
    
    min_people = min(all_labels);
    max_people = max(all_labels);
    
    % 第1步：范围约束
    y_pred_processed = max(min_people, min(max_people, raw_predictions));
    
    % 第2步：基于置信度的自适应阈值
    prediction_confidence = 1 ./ (1 + abs(y_pred_processed - round(y_pred_processed)));
    
    % 第3步：动态阈值计算
    base_threshold = 0.4;
    confidence_adjustment = 0.2 * prediction_confidence;
    dynamic_threshold = base_threshold + confidence_adjustment;
    
    % 第4步：智能取整
    fractional_parts = y_pred_processed - floor(y_pred_processed);
    y_pred_processed = floor(y_pred_processed) + (fractional_parts > dynamic_threshold);
    
    % 第5步：局部一致性校正
    if length(y_pred_processed) > 10
        for i = 6:(length(y_pred_processed)-5)
            local_window = y_pred_processed(i-5:i+5);
            local_mode = mode(local_window);
            
            % 如果当前预测与局部模式差异较大，进行校正
            if abs(y_pred_processed(i) - local_mode) > 2
                confidence_penalty = 1 - prediction_confidence(i);
                if confidence_penalty > 0.3  % 低置信度时进行校正
                    y_pred_processed(i) = local_mode;
                end
            end
        end
    end
    
    % 第6步：最终范围检查
    y_pred_processed = max(min_people, min(max_people, y_pred_processed));
end