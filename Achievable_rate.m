clear
load('theta_self_sup_64beams_I2.mat')

code_num = size(codebook,2);
channel = zeros(length(train_inp),code_num);
W = 1/sqrt(size(codebook,1))*exp(1j*codebook);
for i = 1:(code_num)
    channel(:,i) = train_inp(:,i) + 1j*train_inp(:,i + code_num);
end
SNR = max(abs(W*channel.').^2);