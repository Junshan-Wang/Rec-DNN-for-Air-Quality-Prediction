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
    m = size(x, 1); % ��������
   
    batchsize = opts.batchsize; %���������������С
    numepochs = opts.numepochs; %����ͬһ�����ĵ�������
    
    errors = zeros(1, numepochs);   % ÿ�ε������������������ķ������

    % ��������д��batchsize����Ҫ��m������numbatchesΪ(m/batchsize)����
    % ��������numbatches��ѭ���У����һ�Σ���numbatces�Σ��н�kk�������������ȫ��ȡ��������������[batchsize, 2*batchsize - 1]֮�䣩
    % ��������ѵ������ batch_x = x(kk((numbatches - 1) * batchsize + 1 : end), :);
    %numbatches = m / batchsize;
    numbatches = floor(m / batchsize); % ������������������������С������ȡ�����õ�������

    %assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');
    %L = zeros(numepochs*numbatches);   %^����Ҫ�÷������ڴ濪��̫��
    L = zeros(numepochs*numbatches, 1); % ÿ������������ķ������
    
	n=1; % ������
	
    for i = 1 : numepochs % ��ÿ�ε���
        tic;

        % erL=0;
        kk = randperm(m); % ����һ��������Ϊ1��m��������У�mΪ��������
        nn.learningRate=nn.learningRate*nn.scaling_learningRate;
        
        for l = 1 : numbatches % ��ÿһ��������
            %�������ݿ�
            %�������һ��
            if(l == numbatches) 
                % �������batchsize�������ϳ�һ��������
                batch_x = x(kk((numbatches - 1) * batchsize + 1 : end), :);
                batch_y = y(kk((numbatches - 1) * batchsize + 1 : end), :);
            else
                % ���ѡȡbatchsize��������ֵ��batch_x��batch_y
                batch_x = x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
                batch_y = y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
                %batch_x = x(((l - 1) * batchsize + 1) : (l * batchsize), :);
            end
            % ����������
            %Add noise to input (for use in denoising autoencoder)
            if(nn.inputZeroMaskedFraction ~= 0)
                batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
            end
            
            % �ݶ��½������ƫ����������W��b
            % ����nnff����������nn����һ��ǰ�����
            nn = nnff(nn, batch_x, batch_y); % ǰ������������������ļ���ֵ�����
            nn = nnbp(nn);     % �����ݶ��½��㷨���W��b��ƫ����
            nn = nnapplygrads(nn);  % ����nnbp�����ƫ��������W��b
            
            if nn.sae==1
                tempW=(nn.W{1,1}+nn.W{1,2}')/2; % ������
                nn.W{1,1}=tempW;
                nn.W{1,2}=tempW';
%                 tempb=(nn.b{1,1}+nn.b{1,2})/2;
%                 nn.b{1,1}=tempb;
%                 nn.b{1,2}=tempb;
            end
            % �ô�����������ķ������
            L(n) = nn.L;
            n = n + 1;
        end

        t = toc;
        if(opts.silent ~= 1)
        % if(rem(i, 10) == 0)
          disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1))))]);
          % ����һ�Σ�������������������۾�ֵ
          errors(i) = mean(L((n-numbatches):(n-1)));
%           plot(errors(1:i));
%           drawnow;
        end
    end
end

