function confidence = calculate_prediction_confidence(raw_predictions, processed_predictions)
% 计算预测置信度
prediction_stability = 1 / (1 + std(raw_predictions));
processing_consistency = mean(1 ./ (1 + abs(raw_predictions - processed_predictions)));
confidence = (prediction_stability + processing_consistency) / 2 * 100;
end