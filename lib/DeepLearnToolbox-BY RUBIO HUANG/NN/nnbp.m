function nn = nnbp(nn)
% ���÷��򴫲��㷨��ƫ����
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weight 
% and bias gradients (nn.dW and nn.db)
    
    n = nn.n; % ���������
    sparsityError = 0; %ϡ��������
    
    % d�����ʾ�������Ԫ�Ĳв�
    % d{n}��ʾ�����n��ÿ����Ԫ�Ĳв�
    d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n})); 
    for i = (n - 1) : -1 : 2 % begin:step:end
        if(nn.nonSparsityPenalty>0) %��Ҫ����ϡ�������ƣ��൱��beta
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1); % ʹÿ����������Ӧһ��ƽ�����������
            % �������������Բ��������ʵ��ƽ������ȣ���Ӧ�ʼǵ�3ҳ��ʽ
            sparsityError = nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi));
        end
        d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* (nn.a{i} .* (1 - nn.a{i}));
    end
    % ��������ƫ����
    for i = 1 : (n - 1)
        nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
        nn.db{i} = sum(d{i + 1}, 1)' / size(d{i + 1}, 1);
    end
end
