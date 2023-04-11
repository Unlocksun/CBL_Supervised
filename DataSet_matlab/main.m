% 生成数据集并归一化信道信息的数据
clear all
addpath('./MAT functions');
[DeepMIMO_dataset,params] = O1_gen();

data = DeepMIMO_dataset{1}.user;
num_ant = 64;
DeepMIMO_data.channel = zeros(length(data),2*num_ant);
max_h = 0;

for i = 1:length(data)
    temp = max(abs(data{i}.channel));
    if temp > max_h
        max_h = temp;
    end
    DeepMIMO_data.channel(i,:) = [real(data{i}.channel.'),imag(data{i}.channel.')];
end

%归一化
DeepMIMO_data.channel = DeepMIMO_data.channel/max_h;

save('./DeepMIMO_Dataset/O1_60_64beam.mat','DeepMIMO_data','-v7.3');