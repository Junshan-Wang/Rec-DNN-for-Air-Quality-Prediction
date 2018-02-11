function nn = nnsetup(size)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(size) returns an neural network structure with n=numel(size)
% layers, size being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size   = size; %各层的神经元数量
    nn.n      = numel(nn.size); % 神经网络的总层数
    
    nn.learningRate                     = 0.1;    %  learning rate 
    nn.weightPenaltyL2                  = 0;      %  L2 regularization
    nn.nonSparsityPenalty               = 0;      %  Non sparsity penalty
    nn.sparsityTarget                   = 0.05;   %  Sparsity target (sparsity parameter)
    nn.inputZeroMaskedFraction          = 0;      %  Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;      %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;      %  Internal variable. nntest sets this to one.
    nn.scaling_learningRate             = 1;
    nn.sae                              = 0;

    for i = 2 : nn.n
        nn.b{i - 1} = zeros(nn.size(i), 1);   %  biases 初始化各层神经元的偏置项
                                              %  weights
        nn.W{i - 1} = ( rand(nn.size(i), nn.size(i - 1)) - 0.5 ) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1))); % 随机一个nn.size(i)*nn.size(i-1)的矩阵并为其赋初值
        nn.p{i}     = zeros(1, nn.size(i));   %  average activations (for use with sparsity)存储各层的平均激活度
    end
end
