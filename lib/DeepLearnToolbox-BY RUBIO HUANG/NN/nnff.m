function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural nnwork structure with updated
% layer activations, error and sum squared loss (nn.a, nn.e and nn.L)

    n = nn.n;% 神经网络层数
    m = size(x, 1);% 样本个数

    nn.a{1} = x; % 让输入层的激活值等于输入X

    %feedforward pass
    for i = 2 : n
        % sigm的参数为同时构造m个样本的W*x+b
        nn.a{i} = sigm(repmat(nn.b{i - 1}', m, 1) + nn.a{i - 1} * nn.W{i - 1}');
        if(nn.dropoutFraction > 0 && i<n)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.a{i} = nn.a{i}.*(rand(size(nn.a{i}))>nn.dropoutFraction);
            end 
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            % mean(A, 1)按列求均值
            % 第i层每一个神经元在多个样本下的平均激活度
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
    end

    %error and loss
    % 输出误差
    nn.e = y - nn.a{n};
    % 一次批处理产生的方差代价函数
    nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
end
