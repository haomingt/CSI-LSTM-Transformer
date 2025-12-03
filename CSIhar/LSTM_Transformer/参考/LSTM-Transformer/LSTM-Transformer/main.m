%% LSTM+Transformer人数估计系统 
clear; clc; close all;
%% 1.
fprintf('=== LSTM+Transformer人数估计系统  ===\n');

data_files = {'rssi_data1.mat', 'rssi_data2.mat', 'rssi_data3.mat', 'rssi_data4.mat', ...
              'rssi_data5.mat', 'rssi_data6.mat', 'rssi_data7.mat', 'rssi_data8.mat', ...
              'rssi_data9.mat', 'rssi_data10.mat', 'rssi_data11.mat', 'rssi_data12.mat', ...
              'rssi_data13.mat','rssi_data14.mat','rssi_data16.mat'};

people_mapping = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15];

sequence_length = 200;              % 每个序列多少个采样点
min_samples_per_class = 800;       % 数据不足时，数据增强的阈值，每类多少个样本序列
feature_dims = 64;                  % 增加到64维特征
overlap_step = 5;                  % 重叠步长（即滑动窗口）

%% 2. 深度数据分布分析
fprintf('步骤1: 深度数据分布分析与建模...\n');

raw_data_collection = {};
distribution_models = struct();
successful_loads = 0;

for i = 1:length(data_files)
    filename = data_files{i};  
    try
        if exist(filename, 'file')
            loaded_data = load(filename);
            
            field_names = fieldnames(loaded_data);
            possible_fields = {'rssi', 'data', 'rssi_data', 'signal', 'measurements'};
            
            rssi_data = [];
            for field = possible_fields
                if isfield(loaded_data, field{1})
                    rssi_data = loaded_data.(field{1});
                    break;
                end
            end
            
            if isempty(rssi_data)
                for field = field_names'
                    if isnumeric(loaded_data.(field{1})) && numel(loaded_data.(field{1})) > 10
                        rssi_data = loaded_data.(field{1});
                        break;
                    end
                end
            end
            
            if ~isempty(rssi_data) && length(rssi_data) > 10
                successful_loads = successful_loads + 1;
                rssi_data = double(rssi_data(:));
                people_count = people_mapping(i);
                
                % 深度预处理
                rssi_data = advanced_preprocessing(rssi_data);
                
                % 深度分布建模
                dist_model = analyze_deep_distribution(rssi_data, people_count);
                distribution_models.(sprintf('people_%d', people_count)) = dist_model;
                
                raw_data_collection{end+1} = struct('data', rssi_data, 'people', people_count, 'dist_model', dist_model);
                
                fprintf('✓ 成功分析 %s: 人数%d, 长度=%d, 分布复杂度=%.3f\n', ...
                    filename, people_count, length(rssi_data), dist_model.complexity_score);
            else
                fprintf('⚠ %s 数据格式不正确\n', filename);
            end
        else
            fprintf('⚠ 文件 %s 不存在\n', filename);
        end
    catch ME
        fprintf('✗ 文件%s处理失败: %s\n', filename, ME.message);
    end
end

% 如果真实数据不足，生成基于人数规律的高质量模拟数据
if successful_loads < 3
    fprintf('\n生成基于人数规律的高质量模拟数据...\n');
    [sim_data, sim_models] = generate_people_aware_data(people_mapping, sequence_length);
    raw_data_collection = [raw_data_collection, sim_data];
    
    % 合并分布模型
    fields = fieldnames(sim_models);
    for i = 1:length(fields)
        distribution_models.(fields{i}) = sim_models.(fields{i});
    end
end
    
fprintf('数据分布分析完成！共分析 %d 个人数类别\n', length(fieldnames(distribution_models)));

%% 3. 基于分布的超强数据增强
fprintf('\n步骤2: 基于真实分布的超强数据增强...\n');

all_sequences = [];
all_labels = [];
augment_details = struct();

for i = 1:length(raw_data_collection)
    data_info = raw_data_collection{i};
    rssi_data = data_info.data;
    people_count = data_info.people;
    dist_model = data_info.dist_model;
    
    % 创建长序列
    long_sequences = create_long_sequences(rssi_data, sequence_length, overlap_step);
    
    % 计算需要增强的数量
    current_count = size(long_sequences, 1);
    target_count = min_samples_per_class;
    augment_needed = max(0, target_count - current_count);
    
    fprintf('人数%d: 原始序列%d个，需要增强%d个\n', people_count, current_count, augment_needed);
    
    % 基于分布模型的智能增强
    if augment_needed > 0
        augmented_sequences = distribution_based_augmentation(long_sequences, dist_model, augment_needed, sequence_length);
    else
        augmented_sequences = [];
    end
    
    % 合并所有序列
    all_class_sequences = [long_sequences; augmented_sequences];
    all_sequences = [all_sequences; all_class_sequences];
    all_labels = [all_labels; repmat(people_count, size(all_class_sequences, 1), 1)];
    
    % 记录增强详情
    augment_details.(sprintf('people_%d', people_count)).base = current_count;
    augment_details.(sprintf('people_%d', people_count)).augmented = size(augmented_sequences, 1);
    augment_details.(sprintf('people_%d', people_count)).total = size(all_class_sequences, 1);
    
    fprintf('人数%d: 基础%d + 增强%d = 总计%d序列\n', ...
        people_count, current_count, size(augmented_sequences, 1), size(all_class_sequences, 1));
end

fprintf('智能数据增强完成！总序列数: %d，平均每类: %d\n', size(all_sequences, 1), round(size(all_sequences, 1)/length(unique(all_labels))));

%% 4. 维数超级特征工程
fprintf('\n步骤3: n维超级特征工程...\n');

feature_matrix = extract_ultra_features(all_sequences, feature_dims);

% 高级特征标准化与选择
feature_matrix = advanced_feature_normalization(feature_matrix);

% 转换为cell格式
features = cell(size(all_sequences, 1), 1);
for i = 1:size(all_sequences, 1)
    features{i} = feature_matrix(i, :)';
end

fprintf('n维超级特征提取完成！特征向量大小: %dx1\n', length(features{1}));

%% 5. 智能分层数据划分
fprintf('\n步骤4: 智能分层数据划分...\n');

[X_train, X_val, X_test, y_train, y_val, y_test] = stratified_split_enhanced(features, all_labels);

fprintf('增强数据划分: 训练%d | 验证%d | 测试%d\n', length(y_train), length(y_val), length(y_test));

% 检查数据平衡性
unique_labels = unique(all_labels);
for label = unique_labels'
    train_count = sum(y_train == label);
    val_count = sum(y_val == label);
    test_count = sum(y_test == label);
    fprintf('人数%d: 训练%d | 验证%d | 测试%d\n', label, train_count, val_count, test_count);
end

%% 6. 增强LSTM+Transformer架构
fprintf('\n步骤5: 构建增强LSTM+Transformer网络...\n');

input_size = length(features{1});
[layers, options] = create_enhanced_network(input_size, X_val, y_val);

%% 7. 多阶段训练策略
fprintf('开始多阶段训练...\n');
tic;

% 第一阶段：预训练
fprintf('  阶段1: 预训练（较高学习率）...\n');
options.MaxEpochs = 40;
options.InitialLearnRate = 0.005;
net_stage1 = trainNetwork(X_train, y_train, layers, options);

% 第二阶段：精细调优
fprintf('  阶段2: 精细调优（较低学习率）...\n');
options.MaxEpochs = 60;
options.InitialLearnRate = 0.001;
net = trainNetwork(X_train, y_train, net_stage1.Layers, options);

training_time = toc;
fprintf('多阶段训练完成！总耗时: %.1f秒\n', training_time);
%% 7. 学习率热启动+多阶段训练策略  
% fprintf('开始学习率热启动训练...\n');
% tic;
% 
% % 热启动参数配置
% warmup_epochs = 5;              % 热启动轮数
% initial_lr = 0.003;             % 目标学习率
% total_epochs = 80;              % 总训练轮数
% 
% % === 第一阶段：学习率热启动 (0 -> initial_lr) ===
% fprintf('  🔥 阶段1: 学习率热启动 (0 -> %.4f) - %d轮\n', initial_lr, warmup_epochs);
% 
% % 热启动阶段的学习率线性增长
% warmup_lr_schedule = linspace(initial_lr/warmup_epochs, initial_lr, warmup_epochs);
% 
% for warmup_epoch = 1:warmup_epochs
%     current_lr = warmup_lr_schedule(warmup_epoch);
% 
%     % 设置当前轮次的学习率
%     options_warmup = trainingOptions('adam', ...
%         'MaxEpochs', 1, ...
%         'MiniBatchSize', 32, ...
%         'InitialLearnRate', current_lr, ...
%         'LearnRateSchedule', 'none', ...  % 热启动阶段不使用学习率衰减
%         'ValidationData', {X_val, y_val}, ...
%         'ValidationFrequency', 50, ...
%         'L2Regularization', 1e-4, ...
%         'GradientThreshold', 2, ...
%         'Verbose', false, ...  % 减少输出
%         'Shuffle', 'every-epoch', ...
%         'ExecutionEnvironment', 'auto');
% 
%     if warmup_epoch == 1
%         % 第一轮使用原始网络
%         net_warmup = trainNetwork(X_train, y_train, layers, options_warmup);
%     else
%         % 后续轮次继续训练已有网络
%         net_warmup = trainNetwork(X_train, y_train, net_warmup.Layers, options_warmup);
%     end
% 
%     fprintf('    轮次 %d/%d: 学习率 = %.6f\n', warmup_epoch, warmup_epochs, current_lr);
% end
% 
% fprintf('  ✅ 热启动完成！\n');
% 
% % === 第二阶段：正常训练 (cosine衰减) ===
% fprintf('  🚀 阶段2: 正常训练 (余弦衰减学习率) - %d轮\n', total_epochs - warmup_epochs);
% 
% % 使用余弦衰减的训练选项
% options_main = trainingOptions('adam', ...
%     'MaxEpochs', total_epochs - warmup_epochs, ...
%     'MiniBatchSize', 32, ...
%     'InitialLearnRate', initial_lr, ...
%     'LearnRateSchedule', 'piecewise', ...  % 分段衰减
%     'LearnRateDropFactor', 0.5, ...        % 余弦衰减的近似实现
%     'LearnRateDropPeriod', 20, ...         % 每20轮衰减一次
%     'ValidationData', {X_val, y_val}, ...
%     'ValidationFrequency', 10, ...
%     'L2Regularization', 1e-4, ...
%     'GradientThreshold', 2, ...
%     'Verbose', true, ...
%     'VerboseFrequency', 10, ...
%     'Shuffle', 'every-epoch', ...
%     'ValidationPatience', 25, ...  % 增加耐心值
%     'ExecutionEnvironment', 'auto', ...
%     'Plots', 'training-progress');
% 
% % 继续训练热启动后的网络
% net = trainNetwork(X_train, y_train, net_warmup.Layers, options_main);
% 
% training_time = toc;
% fprintf('学习率热启动+训练完成！总耗时: %.1f秒\n', training_time);
% 
% % === 第三阶段：精细调优 (可选) ===
% fprintf('  🎯 阶段3: 精细调优 (超低学习率) - 20轮\n');
% 
%     options_finetune = trainingOptions('adam', ...
%     'MaxEpochs', 20, ...
%     'MiniBatchSize', 16, ...       % 减小批量大小
%     'InitialLearnRate', initial_lr * 0.01, ... % 使用很低的学习率
%     'LearnRateSchedule', 'none', ...    % 固定学习率
%     'ValidationData', {X_val, y_val}, ...
%     'ValidationFrequency', 5, ...
%     'L2Regularization', 1e-5, ...       % 减少正则化
%     'GradientThreshold', 1, ...        % 更严格的梯度裁剪
%     'Verbose', true, ...
%     'VerboseFrequency', 5, ...        
%     'Shuffle', 'every-epoch', ...
%     'ValidationPatience', 10, ...
%     'ExecutionEnvironment', 'auto');
% 
% 
% net = trainNetwork(X_train, y_train, net.Layers, options_finetune);

final_training_time = toc;
fprintf('🎉 完整学习率热启动训练流程完成！总耗时: %.1f秒\n', final_training_time);

%% 8. 超强智能预测与后处理
fprintf('执行超强智能预测...\n');

% 预测
raw_pred_train = predict(net, X_train);
raw_pred_val = predict(net, X_val);
raw_pred_test = predict(net, X_test);

% 超强智能后处理
y_pred_train = ultra_intelligent_postprocess(raw_pred_train, all_labels, y_train);
y_pred_val = ultra_intelligent_postprocess(raw_pred_val, all_labels, y_val);
y_pred_test = ultra_intelligent_postprocess(raw_pred_test, all_labels, y_test);

%% 8.5 按人数分类详细预测结果输出
fprintf('\n🔍 ================= 按人数分类详细预测结果 =================\n');

unique_people = unique(y_test);
total_correct = 0;
total_samples = 0;

for people_num = unique_people'
    % 找到当前人数的所有样本
    idx = find(y_test == people_num);
    
    if ~isempty(idx)
        % 选择要显示的样本数量（最多100个）
        display_count = min(100, length(idx));
        display_idx = idx(1:display_count);
        
        fprintf('\n人数%d类别 (共%d个样本，显示前%d个):\n', people_num, length(idx), display_count);
        fprintf('序号\t真实人数\t预测人数\t原始预测\t误差\n');
        fprintf('------------------------------------------------\n');
        
        class_correct = 0;
        
        for i = 1:display_count
            sample_idx = display_idx(i);
            raw_val = raw_pred_test(sample_idx);
            pred_val = y_pred_test(sample_idx);
            true_val = y_test(sample_idx);
            error_val = true_val - pred_val;
            
            % 统计正确预测
            if abs(error_val) <= 0.5  % 四舍五入后正确
                class_correct = class_correct + 1;
            end
            
            fprintf('%d\t\t%d\t\t%.1f\t\t%.3f\t\t%+.1f\n', ...
                sample_idx, true_val, pred_val, raw_val, error_val);
        end
        
        fprintf('------------------------------------------------\n');
        fprintf('人数%d准确率: %.1f%% (%d/%d)\n\n', ...
            people_num, class_correct/display_count*100, class_correct, display_count);
        
        total_correct = total_correct + class_correct;
        total_samples = total_samples + display_count;
    end
end

fprintf('========================================================\n');
fprintf('总体显示样本准确率: %.1f%% (%d/%d)\n', ...
    total_correct/total_samples*100, total_correct, total_samples);
fprintf('========================================================\n');
%% 9. 综合性能评估
fprintf('\n步骤6: 综合性能评估...\n');

train_metrics = calculate_comprehensive_metrics(y_train, y_pred_train);
val_metrics = calculate_comprehensive_metrics(y_val, y_pred_val);
test_metrics = calculate_comprehensive_metrics(y_test, y_pred_test);

print_enhanced_performance_report(train_metrics, val_metrics, test_metrics);

%% 10. 超级可视化分析
fprintf('\n步骤7: 生成超级可视化分析...\n');

create_ultra_visualization(y_test, y_pred_test, train_metrics, val_metrics, test_metrics, ...
                          augment_details, all_labels, training_time, distribution_models);

%% 11. 保存与最终报告
save_enhanced_results(net, options, distribution_models, augment_details, ...
                     train_metrics, val_metrics, test_metrics, training_time);