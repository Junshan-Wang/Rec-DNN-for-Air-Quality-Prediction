  function [nn, L] = nntrain(nn, x, y, opts)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and 
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum 
% squared error for each training minibatch.
% figure;
    if(~isfield(opts, 'silent'))
        opts.silent = 0;
    end

    assert(isfloat(x), 'x must be a float');
    m = size(x, 1); % 样本容量
   
    batchsize = opts.batchsize; %设置批处理的批大小
    numepochs = opts.numepochs; %设置同一样本的迭代次数
    
    errors = zeros(1, numepochs);   % 每次迭代，所有样本产生的方差代价

    % ！！！改写，batchsize不需要被m整除，numbatches为(m/batchsize)的商
    % ！！！在numbatches次循环中，最后一次（第numbatces次）中将kk后面的所有数据全部取出（样本数量在[batchsize, 2*batchsize - 1]之间）
    % ！！！并训练，即 batch_x = x(kk((numbatches - 1) * batchsize + 1 : end), :);
    %numbatches = m / batchsize;
    numbatches = floor(m / batchsize); % 批容量：样本总数除以批大小，再下取整，得到批容量

    %assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');
    %L = zeros(numepochs*numbatches);   %^不需要用方阵，且内存开销太大
    L = zeros(numepochs*numbatches, 1); % 每次批处理产生的方差代价
    
	n=1; % 计数器
	
    for i = 1 : numepochs % 对每次迭代
        tic;

        % erL=0;
        kk = randperm(m); % 返回一个行向量为1到m的随机排列，m为样本容量
        nn.learningRate=nn.learningRate*nn.scaling_learningRate;
        
        for l = 1 : numbatches % 对每一次批处理
            %构造数据块
            %若是最后一次
            if(l == numbatches) 
                % 将最后不足batchsize个样本合成一批做处理
                batch_x = x(kk((numbatches - 1) * batchsize + 1 : end), :);
                batch_y = y(kk((numbatches - 1) * batchsize + 1 : end), :);
            else
                % 随机选取batchsize个样本赋值给batch_x和batch_y
                batch_x = x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
                batch_y = y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
                %batch_x = x(((l - 1) * batchsize + 1) : (l * batchsize), :);
            end
            % ？？？？？
            %Add noise to input (for use in denoising autoencoder)
            if(nn.inputZeroMaskedFraction ~= 0)
                batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
            end
            
            % 梯度下降法求解偏导数并更新W，b
            % 调用nnff函数对网络nn进行一次前向计算
            nn = nnff(nn, batch_x, batch_y); % 前馈计算求解神经网络各层的激活值和误差
            nn = nnbp(nn);     % 利用梯度下降算法求解W和b的偏导数
            nn = nnapplygrads(nn);  % 利用nnbp所求的偏导数更新W和b
            
            if nn.sae==1
                tempW=(nn.W{1,1}+nn.W{1,2}')/2; % ？？？
                nn.W{1,1}=tempW;
                nn.W{1,2}=tempW';
%                 tempb=(nn.b{1,1}+nn.b{1,2})/2;
%                 nn.b{1,1}=tempb;
%                 nn.b{1,2}=tempb;
            end
            % 该次批处理产生的方差代价
            L(n) = nn.L;
            n = n + 1;
        end

        t = toc;
        if(opts.silent ~= 1)
        % if(rem(i, 10) == 0)
          disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1))))]);
          % 迭代一次，所有样本产生方差代价均值
          errors(i) = mean(L((n-numbatches):(n-1)));
%           plot(errors(1:i));
%           drawnow;
        end
    end
end

