% function [layers, options] = create_enhanced_network(input_size, X_val, y_val)
%     % 创建增强的LSTM+Transformer网络
% 
%     lstm_units = 256;      % 增加LSTM单元
%     dense_units = 128;     % 增加全连接层单元
%     dropout_rate = 0.3;    % 适中的dropout

    % layers = [
    %     % 输入层
    %     sequenceInputLayer(input_size, 'Name', 'input', 'MinLength', 1)
    % 
    %     % 第一层LSTM编码器
    %     lstmLayer(lstm_units, 'OutputMode', 'sequence', 'Name', 'lstm_encoder1')
    %     batchNormalizationLayer('Name', 'bn_encoder1')
    %     dropoutLayer(dropout_rate, 'Name', 'dropout_encoder1')
    % 
    %     % 第二层LSTM编码器（注意力模拟）
    %     lstmLayer(round(lstm_units*0.75), 'OutputMode', 'sequence', 'Name', 'lstm_encoder2')
    %     batchNormalizationLayer('Name', 'bn_encoder2')
    %     dropoutLayer(dropout_rate, 'Name', 'dropout_encoder2')
    % 
    %     % 第三层LSTM（深度特征提取）
    %     lstmLayer(round(lstm_units*0.5), 'OutputMode', 'sequence', 'Name', 'lstm_encoder3')
    %     batchNormalizationLayer('Name', 'bn_encoder3')
    %     dropoutLayer(dropout_rate, 'Name', 'dropout_encoder3')
    % 
    %     % 注意力机制模拟（全局池化）
    %     globalMaxPooling1dLayer('Name', 'global_attention')
    % 
    %     % 深度特征融合网络
    %     fullyConnectedLayer(dense_units*2, 'Name', 'fusion_dense1')
    %     batchNormalizationLayer('Name', 'bn_fusion1')
    %     reluLayer('Name', 'relu_fusion1')
    %     dropoutLayer(dropout_rate, 'Name', 'dropout_fusion1')
    % 
    %     fullyConnectedLayer(dense_units, 'Name', 'fusion_dense2')
    %     batchNormalizationLayer('Name', 'bn_fusion2')
    %     reluLayer('Name', 'relu_fusion2')
    %     dropoutLayer(dropout_rate, 'Name', 'dropout_fusion2')
    % 
    %     fullyConnectedLayer(round(dense_units*0.5), 'Name', 'fusion_dense3')
    %     batchNormalizationLayer('Name', 'bn_fusion3')
    %     reluLayer('Name', 'relu_fusion3')
    %     dropoutLayer(dropout_rate*0.7, 'Name', 'dropout_fusion3')
    % 
    %     fullyConnectedLayer(round(dense_units*0.25), 'Name', 'pre_output1')
    %     batchNormalizationLayer('Name', 'bn_pre_output1')
    %     reluLayer('Name', 'relu_pre_output1')
    %     dropoutLayer(dropout_rate*0.5, 'Name', 'dropout_pre_output1')
    % 
    %     fullyConnectedLayer(16, 'Name', 'pre_output2')
    %     reluLayer('Name', 'relu_pre_output2')
    %     dropoutLayer(dropout_rate*0.3, 'Name', 'dropout_pre_output2')
    % 
    %     % 输出层
    %     fullyConnectedLayer(1, 'Name', 'output')
    %     regressionLayer('Name', 'regression')
    % ];



function [layers, options] = create_enhanced_network(input_size, X_val, y_val)
    % LSTM -> (Projection + Positional Encoding) -> TransformerEncoder x N
    % -> GlobalAvgPool -> Dense head -> Regression

    % --------- 超参数（与你给定的保持一致） ---------
    lstm_units   = 256;     % LSTM 单元数
    dense_units  = 128;     % 全连接基准宽度
    dropout_rate = 0.30;    % Dropout

    % --------- Transformer 相关参数（可按需微调） ---------
    dModel      = 128;      % Transformer 通道维度
    numHeads    = 4;        % 注意力头数（需整除 dModel）
    ffnDim      = 256;      % FFN 内部宽度（~ 2x dModel）
    numEncoders = 3;        % Encoder 层数（Nx）

    % --------- 主体网络定义 ---------
    % 注意：positionalEncoding1dLayer/transformerEncoderLayer 需要 R2023b 或更新
    try
        layers = [
            % 输入（每个时间步 input_size 维）
            sequenceInputLayer(input_size, 'Name','input', 'MinLength', 1)

            % 前端 LSTM 编码（噪声抑制 + 局部时序建模）
            lstmLayer(lstm_units, 'OutputMode','sequence', 'Name','lstm1')
            dropoutLayer(dropout_rate, 'Name','dropout_lstm1')

            % 线性投影到 dModel（与 Transformer 通道对齐）
            fullyConnectedLayer(dModel, 'Name','proj_to_dModel')

            % 位置编码（与第二张图一致）
            positionalEncoding1dLayer(dModel, 'Name','posenc')

            % Transformer Encoder × N（封装了 MHA、FFN、Add&Norm）
            transformerEncoderLayer(numHeads, dModel, ffnDim, ...
                'Name','encoder1', 'DropoutFactor', dropout_rate)

            transformerEncoderLayer(numHeads, dModel, ffnDim, ...
                'Name','encoder2', 'DropoutFactor', dropout_rate)

            transformerEncoderLayer(numHeads, dModel, ffnDim, ...
                'Name','encoder3', 'DropoutFactor', dropout_rate)

            % 序列汇聚（也可换 globalMaxPooling1dLayer）
            globalAveragePooling1dLayer('Name','pool')

            % 输出头（两层 MLP）
            fullyConnectedLayer(dense_units, 'Name','head_fc1')
            reluLayer('Name','head_relu1')
            dropoutLayer(dropout_rate, 'Name','head_dropout1')

            fullyConnectedLayer(round(dense_units*0.5), 'Name','head_fc2')
            reluLayer('Name','head_relu2')
            dropoutLayer(dropout_rate*0.5, 'Name','head_dropout2')

            fullyConnectedLayer(1, 'Name','output')
            regressionLayer('Name','regression')
        ];
    catch
        % ---- 降级方案（老版本没有 Transformer 层时） ----
        % 用自注意力/全局池化做简化，结构仍保持“串联”思想
        warning('transformerEncoderLayer/positionalEncoding1dLayer unavailable. Falling back to simplified attention head.');

        layers = [
            sequenceInputLayer(input_size, 'Name','input', 'MinLength', 1)

            lstmLayer(lstm_units, 'OutputMode','sequence', 'Name','lstm1')
            dropoutLayer(dropout_rate, 'Name','dropout_lstm1')

            % 简化注意力：用 1D 自注意力可替代；若也没有，先用全局平均池化
            %（根据你版本选择其一；此处用全局平均池化以保证可运行）
            globalAveragePooling1dLayer('Name','pool_simplified')

            fullyConnectedLayer(dense_units, 'Name','head_fc1')
            reluLayer('Name','head_relu1')
            dropoutLayer(dropout_rate, 'Name','head_dropout1')

            fullyConnectedLayer(round(dense_units*0.5), 'Name','head_fc2')
            reluLayer('Name','head_relu2')
            dropoutLayer(dropout_rate*0.5, 'Name','head_dropout2')

            fullyConnectedLayer(1, 'Name','output')
            regressionLayer('Name','regression')
        ];
    end






     % layers = [    %单纯LSTM
     %            sequenceInputLayer(input_size, 'Name','input', 'MinLength', 1)
     % 
     %            lstmLayer(lstm_units, 'OutputMode','sequence', 'Name','lstm1')
     %            batchNormalizationLayer('Name','bn1')
     %            dropoutLayer(dropout_rate, 'Name','drop1')
     % 
     %            lstmLayer(round(lstm_units*0.75), 'OutputMode','sequence', 'Name','lstm2')
     %            batchNormalizationLayer('Name','bn2')
     %            dropoutLayer(dropout_rate, 'Name','drop2')
     % 
     %            % 第三层直接 'last'：输出 [C,B]，省去池化/注意力
     %            lstmLayer(round(lstm_units*0.5), 'OutputMode','last', 'Name','lstm3_last')
     % 
     %            fullyConnectedLayer(dense_units*2, 'Name','fc1')
     %            batchNormalizationLayer('Name','bn3')
     %            reluLayer('Name','relu1')
     %            dropoutLayer(dropout_rate, 'Name','drop3')
     % 
     %            fullyConnectedLayer(dense_units, 'Name','fc2')
     %            batchNormalizationLayer('Name','bn4')
     %            reluLayer('Name','relu2')
     %            dropoutLayer(dropout_rate, 'Name','drop4')
     % 
     %            fullyConnectedLayer(round(dense_units*0.5), 'Name','fc3')
     %            reluLayer('Name','relu3')
     %            dropoutLayer(dropout_rate*0.7, 'Name','drop5')
     % 
     %            fullyConnectedLayer(16, 'Name','fc4')
     %            reluLayer('Name','relu4')
     %            dropoutLayer(dropout_rate*0.3, 'Name','drop6')
     % 
     %            fullyConnectedLayer(1, 'Name','output')
     %            regressionLayer('Name','regression')
     %        ];
    % 参数
%     options = trainingOptions('adam', ...
%         'MaxEpochs', 50, ...
%         'MiniBatchSize', 64, ...
%         'InitialLearnRate', 0.01, ...
%         'LearnRateSchedule', 'piecewise', ...
%         'LearnRateDropFactor', 0.5, ...
%         'LearnRateDropPeriod', 20, ...
%         'ValidationData', {X_val, y_val}, ...
%         'ValidationFrequency', 15, ...
%         'L2Regularization', 1e-4, ...
%         'GradientThreshold', 1, ...
%         'Verbose', true, ...
%         'VerboseFrequency', 5, ...
%         'Shuffle', 'every-epoch', ...
%         'ValidationPatience', 100, ...
%         'ExecutionEnvironment', 'auto', ...
%         'Plots', 'training-progress');
% end


% function [layers, options] = create_enhanced_network(input_size, X_val, y_val)
% % Pure-Transformer regression network built from selfAttentionLayer blocks.
% % input_size : feature dimension per timestep
% % X_val, y_val: validation data
% 
%     % ====== Hyperparameters ======
%     dModel       = 128;      % hidden size (also FFN output size)
%     numHeads     = 8;        % must divide dModel
%     ffnDim       = 4 * dModel;
%     numBlocks    = 3;        % number of encoder blocks
%     dropAttn     = 0.3;
%     dropFFN      = 0.3;
% 
%     % ====== Stem: project input to dModel ======
%     lgraph = layerGraph();
%     stem = [
%         sequenceInputLayer(input_size, 'Name','input', 'MinLength',1)
%         fullyConnectedLayer(dModel, 'Name','proj')
%         layerNormalizationLayer('Name','ln_stem')
%     ];
%     lgraph = addLayers(lgraph, stem);
% 
%     prevName = 'ln_stem';
% 
%     % ====== Encoder Blocks ======
%     for i = 1:numBlocks
%         % names
%         ln1   = sprintf('enc%d_ln1', i);
%         sa    = sprintf('enc%d_sa',  i);
%         drop1 = sprintf('enc%d_drop_sa', i);
%         add1  = sprintf('enc%d_add1', i);
% 
%         ln2   = sprintf('enc%d_ln2', i);
%         f1    = sprintf('enc%d_ffn1', i);
%         relu1 = sprintf('enc%d_relu1', i);
%         drop2 = sprintf('enc%d_drop_ffn1', i);
%         f2    = sprintf('enc%d_ffn2', i);
%         drop3 = sprintf('enc%d_drop_ffn2', i);
%         add2  = sprintf('enc%d_add2', i);
% 
%         block = [
%             layerNormalizationLayer('Name', ln1)
%             selfAttentionLayer(numHeads, dModel, ...
%                 'Name', sa, 'DropoutProbability', dropAttn, ...
%                 'OutputSize', dModel)
%             dropoutLayer(dropAttn, 'Name', drop1)
%             additionLayer(2, 'Name', add1)
% 
%             layerNormalizationLayer('Name', ln2)
%             fullyConnectedLayer(ffnDim, 'Name', f1)
%             reluLayer('Name', relu1)
%             dropoutLayer(dropFFN, 'Name', drop2)
%             fullyConnectedLayer(dModel, 'Name', f2)
%             dropoutLayer(dropFFN, 'Name', drop3)
%             additionLayer(2, 'Name', add2)
%         ];
%         lgraph = addLayers(lgraph, block);
% 
%         % connections: pre-norm + residuals
%         % input -> ln1
%         lgraph = connectLayers(lgraph, prevName, ln1);
%         % skip for add1
%         lgraph = connectLayers(lgraph, prevName, [add1 '/in2']);
%         % MHA path to add1
%         lgraph = connectLayers(lgraph, drop1, [add1 '/in1']);
% 
%         % add1 -> ln2
%         lgraph = connectLayers(lgraph, add1, ln2);
%         % FFN path to add2
%         lgraph = connectLayers(lgraph, drop3, [add2 '/in1']);
%         % skip from add1 to add2
%         lgraph = connectLayers(lgraph, add1, [add2 '/in2']);
% 
%         prevName = add2; % output of block i
%     end
% 
%     % ====== Head ======
%     head = [
%         globalAveragePooling1dLayer('Name','pool')   % stable global aggregation
%         fullyConnectedLayer(256, 'Name','head_fc1')
%         layerNormalizationLayer('Name','head_ln1')
%         reluLayer('Name','head_relu1')
%         dropoutLayer(0.3, 'Name','head_drop1')
% 
%         fullyConnectedLayer(128, 'Name','head_fc2')
%         layerNormalizationLayer('Name','head_ln2')
%         reluLayer('Name','head_relu2')
%         dropoutLayer(0.2, 'Name','head_drop2')
% 
%         fullyConnectedLayer(32, 'Name','pre_output')
%         reluLayer('Name','pre_relu')
%         fullyConnectedLayer(1, 'Name','output')
%         regressionLayer('Name','regression')
%     ];
%     lgraph = addLayers(lgraph, head);
%     lgraph = connectLayers(lgraph, prevName, 'pool');
% 
%     layers = lgraph;

    % ====== Training Options ======
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 20, ...
        'ValidationData', {X_val, y_val}, ...
        'ValidationFrequency', 15, ...
        'L2Regularization', 1e-4, ...
        'GradientThreshold', 1, ...
        'Verbose', true, ...
        'VerboseFrequency', 5, ...
        'Shuffle', 'every-epoch', ...
        'ValidationPatience', 100, ...
        'ExecutionEnvironment', 'auto', ...
        'Plots', 'training-progress');
end

