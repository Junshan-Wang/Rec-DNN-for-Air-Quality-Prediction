%% -----Question-----
%nntrain:
%	line 55 : 如何加噪

%nnff:
%line15:dropoutFraction的作用
%line26:公式

%opts.silent是什么意思



function nn = nnsetup(size)
% 构造神经网络，返回一个n = numel(size)层的神经网络，size为一个n*1的向量
	
	%%-----神经网络变量------
	nn.size = size; %一个n*1的数组，表示各层的神经元数量
	nn.n = numel(nn.size); %神经网络的总层数
	
	nn.learningRate = 0.1; % 学习率
	nn.weightPenaltyL2 = 0; %权重惩罚项
	nn.nonSparisityPenalty = 0; 
	nn.sparisityTarget = 0.05; % 稀疏性限制参数
	nn.inputZeroMaskedFraction = 0; % 用于去噪
	nn.dropoutFraction = 0;
	nn.testing = 0;	
	nn.scaling_learningRate = 1;
	nn.sae = 0; %是否是一个自编码网络
	
	%%---初始化b,W，这两个数组为cell型数组，即每个元素仍为一个数组---
	for i = 2:nn.n
		nn.b{i-1} = zeros(nn.size(i), 1); %初始化各层的偏置值
		%初始化各层的权重值
		nn.W{i-1} = ( rand(nn.size(i), nn.size(i-1)) - 0.5 ) * 2 * 4 * sqrt(6/(nn.size(i)+nn.size(i-1)));
		nn.p{i} = zeros(1, nn.size(i)); % 各层单个神经元在多个样本下的平均激活度
	end
end


%% ---前馈传导计算，为神经网络新增参数nn.a和nn.e-----
function [nn, L] = nnff(nn, x, y, opts)
% nn :被训练的神经网络
% 参数含义：  x:输入 y:输出 opts.numepochs: 迭代次数 opts.batchsize: 批大小
% 返回值：  nn: nn.a(更新的激活值)，nn.e（误差），nn.W(权重)，nn.b(偏置)



%% ----反向传播计算，为神经网络新增参数nn.dW和nn.db,即每条边的Wb所对应的偏导数
function nn = nnbp(nn)
% 利用反向传播算法求偏导数


%% ---- 用nnbp求得的偏导数更新W b
function nn = nnapplygrads(nn)




function [nn, L] = nntrain(nn, x, y, opts)
% 训练一个神经网络

