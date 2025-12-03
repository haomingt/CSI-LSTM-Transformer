  load('rssi_data_test1.mat', 'rssi_data_test1');
 predicted_count = predict_people_count(rssi_data_test1);
fprintf('预测人数: %d\n', predicted_count);

