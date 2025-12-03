function predicted_people = predict_people_count(new_rssi_data, model_path)
% 使用训练好的模型预测人数
% 输入：
%   new_rssi_data: 新的RSSI数据 (向量)
%   model_path: 模型文件路径 (可选，默认为当前目录)
% 输出：
%   predicted_people: 预测的人数

%% 1. 参数设置
if nargin < 2
    model_path = './';  % 默认当前目录
end

% 模型参数（必须与训练时一致）
sequence_length = 300;
feature_dims = 64;

%% 2. 加载训练好的模型
try
    fprintf('正在加载模型...\n');
    model_file = fullfile(model_path, 'enhanced_lstm_transformer_model.mat');
    load(model_file, 'net');
    fprintf('✓ 模型加载成功\n');
catch ME
    error('❌ 模型加载失败: %s\n请确保模型文件存在于: %s', ME.message, model_file);
end

%% 3. 数据预处理
fprintf('正在预处理新数据...\n');
try
    processed_data = advanced_preprocessing(new_rssi_data);
    fprintf('✓ 数据预处理完成\n');
catch ME
    error('❌ 数据预处理失败: %s', ME.message);
end

%% 4. 创建序列
fprintf('正在创建预测序列...\n');
try
    sequences = create_prediction_sequences(processed_data, sequence_length);
    fprintf('✓ 创建了 %d 个预测序列\n', size(sequences, 1));
catch ME
    error('❌ 序列创建失败: %s', ME.message);
end
% % %% 4.5 预测数据增强
% % fprintf('正在进行预测数据增强...\n');
% % try
% %     % 计算需要增强的数量
% %     current_seq_count = size(sequences, 1);
% %     target_seq_count = max(20, current_seq_count * 3);  % 至少20个序列，或3倍增强
% %     augment_needed = target_seq_count - current_seq_count;
% % 
% %     if augment_needed > 0
% %         augmented_sequences = prediction_data_augmentation(sequences, augment_needed, sequence_length);
% %         sequences = [sequences; augmented_sequences];
% %         fprintf('✓ 数据增强完成：原始%d个 + 增强%d个 = 总计%d个序列\n', ...
% %             current_seq_count, size(augmented_sequences, 1), size(sequences, 1));
% %     else
% %         fprintf('✓ 序列数量充足，跳过数据增强\n');
% %     end
% % catch ME
% %     fprintf('⚠️  数据增强失败，使用原始序列: %s\n', ME.message);
% % end
%% 5. 特征提取
fprintf('正在提取特征...\n');
try
    features = extract_prediction_features(sequences, feature_dims);
    fprintf('✓ 特征提取完成，特征维度: %dx%d\n', size(features, 1), size(features, 2));
catch ME
    error('❌ 特征提取失败: %s', ME.message);
end

%% 6. 模型预测
fprintf('正在进行预测...\n');
try
    % 转换为cell格式
    features_cell = cell(size(features, 1), 1);
    for i = 1:size(features, 1)
        features_cell{i} = features(i, :)';
    end
    
    % 使用模型预测
    raw_predictions = predict(net, features_cell);
    fprintf('✓ 原始预测完成\n');
catch ME
    error('❌ 模型预测失败: %s', ME.message);
end

%% 7. 后处理
fprintf('正在后处理预测结果...\n');
try
    processed_predictions = intelligent_postprocess_single(raw_predictions);
    
    % 取多个序列预测的平均值/众数
    if length(processed_predictions) > 1
        predicted_people = round(median(processed_predictions));
    else
        predicted_people = processed_predictions;
    end
    
    fprintf('✓ 预测完成！\n');
catch ME
    error('❌ 后处理失败: %s', ME.message);
end

%% 8. 输出结果
fprintf('\n🎯 ================= 预测结果 =================\n');
fprintf('输入数据长度: %d\n', length(new_rssi_data));
fprintf('生成序列数量: %d\n', size(sequences, 1));
fprintf('原始预测范围: %.2f - %.2f\n', min(raw_predictions), max(raw_predictions));
fprintf('后处理预测范围: %.1f - %.1f\n', min(processed_predictions), max(processed_predictions));
fprintf('最终预测人数: %d\n', predicted_people);
fprintf('预测置信度: %.1f%%\n', calculate_prediction_confidence(raw_predictions, processed_predictions));
fprintf('============================================\n');

end