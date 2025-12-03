function processed_predictions = intelligent_postprocess_single(raw_predictions)
% 智能后处理（单次预测版本）
min_people = 1;   % 假设最小人数为1
max_people = 13;  % 假设最大人数为13

% 范围约束
processed_predictions = max(min_people, min(max_people, raw_predictions));

% 自适应阈值处理
confidence_scores = 1 ./ (1 + abs(processed_predictions - round(processed_predictions)));
base_threshold = 0.4;
confidence_adjustment = 0.2 * confidence_scores;
dynamic_threshold = base_threshold + confidence_adjustment;

% 智能取整
fractional_parts = processed_predictions - floor(processed_predictions);
processed_predictions = floor(processed_predictions) + (fractional_parts > dynamic_threshold);

% 最终范围检查
processed_predictions = max(min_people, min(max_people, processed_predictions));
end