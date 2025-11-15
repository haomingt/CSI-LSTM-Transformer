
function print_enhanced_performance_report(train_metrics, val_metrics, test_metrics)
    % 输出增强的性能报告
    datasets = {'训练集', '验证集', '测试集'};
    metrics_sets = {train_metrics, val_metrics, test_metrics};
    
    fprintf('\n🎯 ==================== 性能评估报告 ====================\n');
    
    for i = 1:3
        metrics = metrics_sets{i};
        fprintf('\n=== %s 详细评估结果 ===\n', datasets{i});
        fprintf('📊 回归指标:\n');
        fprintf('   RMSE: %.4f | MAE: %.4f | R²: %.4f | 相关性: %.4f\n', ...
            metrics.rmse, metrics.mae, metrics.r2, metrics.correlation);
        
        fprintf('🎯 准确率指标:\n');
        fprintf('   精确率: %.1f%% | ±1容忍: %.1f%% | ±2容忍: %.1f%%\n', ...
            metrics.exact_acc, metrics.tolerance1_acc, metrics.tolerance2_acc);
        
        fprintf('📈 分类指标:\n');
        fprintf('   平均精确度: %.3f | 平均召回率: %.3f | F1分数: %.3f\n', ...
            metrics.avg_precision, metrics.avg_recall, metrics.f1_score);
        
        fprintf('⚡ 稳定性指标:\n');
        fprintf('   误差标准差: %.4f | 最大误差: %.2f | MAPE: %.2f%%\n', ...
            metrics.error_std, metrics.max_error, metrics.mape);
    end
    
    % 性能等级评估
    fprintf('\n🏆 ==================== 性能等级评估 ====================\n');
    test_acc = test_metrics.exact_acc;
    test_tolerance1 = test_metrics.tolerance1_acc;
    
    if test_acc >= 80
        fprintf('🥇 金牌级性能！精确率达到 %.1f%%\n', test_acc);
    elseif test_acc >= 70
        fprintf('🥈 银牌级性能！精确率达到 %.1f%%\n', test_acc);
    elseif test_acc >= 60
        fprintf('🥉 铜牌级性能！精确率达到 %.1f%%\n', test_acc);
    else
        fprintf('⚠️  性能需要优化，当前精确率: %.1f%%\n', test_acc);
    end
    
    if test_tolerance1 >= 95
        fprintf('✨ ±1容忍度表现优异: %.1f%%\n', test_tolerance1);
    elseif test_tolerance1 >= 85
        fprintf('👍 ±1容忍度表现良好: %.1f%%\n', test_tolerance1);
    else
        fprintf('📈 ±1容忍度有提升空间: %.1f%%\n', test_tolerance1);
    end
end
