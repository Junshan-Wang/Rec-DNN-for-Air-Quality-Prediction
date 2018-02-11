function nn = nnbp(nn)
% 利用反向传播算法求偏导数
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weight 
% and bias gradients (nn.dW and nn.db)
    
    n = nn.n; % 神经网络层数
    sparsityError = 0; %稀疏性限制
    
    % d数组表示各层的神经元的残差
    % d{n}表示输出层n的每个神经元的残差
    d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n})); 
    for i = (n - 1) : -1 : 2 % begin:step:end
        if(nn.nonSparsityPenalty>0) %需要加入稀疏性限制，相当于beta
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1); % 使每个样本都对应一个平均激活度向量
            % 用期望的抑制性参数点除以实际平均激活度，对应笔记第3页公式
            sparsityError = nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi));
        end
        d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* (nn.a{i} .* (1 - nn.a{i}));
    end
    % 计算各层的偏导数
    for i = 1 : (n - 1)
        nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
        nn.db{i} = sum(d{i + 1}, 1)' / size(d{i + 1}, 1);
    end
end
