clear all;
clc;

load air_bj.mat;

station=1;

numTimeDelay=0:0;

Epochs=50;
Batchsize=10;

%%inputPolluents=[1 2 3 4 5 6]; 
inputPolluents=[1 2 3 4 5 6];     %�����ȥ��Ⱦ������
outputPolluents=6;                  %���PM2.5

weatherFactors=[7 8 9 10];      %����δ������Ԥ��

predict=[];
test=[];
%%%%%%%%%%%%%%%%%Ԥ��������Ժ�%%%%%%%%%%%%%%%%%%%%
for iTimeDelay=1:length(numTimeDelay);

timeDelay=numTimeDelay(iTimeDelay);    %Ԥ�⼸Сʱ֮���ֵ��0Ϊһ��
range=timeDelay+1;

%inputSize=range*length(inputPolluents)+3*length(weatherFactors);    
inputSize=range*length(inputPolluents)+range*length(weatherFactors);    
outputSize=1;

trainingLength=2400;
testingLength=240;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%���������������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_x=zeros(trainingLength,inputSize);
train_y=zeros(trainingLength,outputSize);

for i=1:trainingLength
    t=[];
    for j=1:length(inputPolluents)
        %ǰrange��Ŀ�������
        t=[t air_bj{station,1}(i:i+range-1,inputPolluents(j))'];
    end
    for j=1:length(weatherFactors)
        %��range�������Ԥ��
        %t=[t mean(air_bj{station,1}(i+range:i+range+timeDelay,weatherFactors(j))) max(air_bj{station,1}(i+range:i+range+timeDelay,weatherFactors(j))) air_bj{station,1}(i+range+timeDelay,weatherFactors(j))];
        t=[t air_bj{station,1}(i+range:i+range+timeDelay,weatherFactors(j))'];
    end
    train_x(i,:)=t;
    
    train_y(i,1)=air_bj{station,1}(i+range+timeDelay,outputPolluents);
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hiddenLayers=2;     %���ز���

hiddenNodes1=50;    %��һ��������
hiddenNodes2=30;    %�ڶ���������
hiddenStruct=[hiddenNodes1 hiddenNodes2];

%%%%%%%%%sae��ȡ����%%%%%%%%%%%
sae=saesetup([inputSize hiddenStruct]);
for iSae=1:numel(hiddenStruct)
    sae.ae{iSae}.learningRate=1;
    sae.ae{iSae}.scaling_learningRate=0.99;
    sae.ae{iSae}.sae=1;
end
opts.numepochs = Epochs;
opts.batchsize = Batchsize;
opts.silent = 1;
sae=saetrain(sae,train_x,opts);

%%%%ÿһ��������ȡ���ƫ�ú�Ȩ�ؽ��fnn,�õ�train_feature%%%%
fnn=nnsetup([inputSize hiddenStruct]);
for iFnn=1:numel(hiddenStruct)
    fnn.W{iFnn}=sae.ae{iFnn}.W{1};
    fnn.b{iFnn}=sae.ae{iFnn}.b{1};
end
fnn=nnff(fnn,train_x,zeros(length(train_x),hiddenStruct(end)));
train_feature=fnn.a{end};



%%%%%%%��ͨ������nn�ع�Ԥ��%%%%%%%%%
nn=nnsetup([hiddenStruct(end) outputSize]);
opts.sae=0;
opts.numepochs = Epochs;
opts.batchsize = Batchsize;
opts.silent = 1;
nn.learningRate = 1;
nn.scaling_learningRate = 0.99;
nn = nntrain(nn, train_feature, train_y, opts); 

%%%%ջʽ�Ա������ͨ������%%%%%%
dnn=nnsetup([inputSize hiddenStruct outputSize]);
for iDnn=1:numel(hiddenStruct)
    dnn.W{iDnn}=fnn.W{iDnn};
    dnn.b{iDnn}=fnn.b{iDnn};
end
dnn.W{numel(hiddenStruct)+1}=nn.W{1};
dnn.b{numel(hiddenStruct)+1}=nn.b{1};

opts.sae=0;
opts.numepochs = Epochs;
opts.batchsize = Batchsize;
opts.silent = 1;
dnn.learningRate = 1;
dnn.scaling_learningRate = 0.99;

%dnn=nntrain(dnn,train_x,train_y,opts);



%%%%%%%%%%%%%%%%%%%%�ع����%%%%%%%%%%%%%%%%%%%%%
rhiddenStruct=[hiddenNodes2 hiddenNodes1];

rfnn=nnsetup([rhiddenStruct inputSize]);
for iFnn=1:numel(rhiddenStruct)
    rfnn.W{iFnn}=sae.ae{end+1-iFnn}.W{2};
    rfnn.b{iFnn}=sae.ae{end+1-iFnn}.b{2};
end

%{
dnn=nnff(dnn,train_x,train_y);

retrain_x=dnn.a{end-1};
retrain_y=dnn.a{1};

opts.sae=0;
opts.numepochs = Epochs;
opts.batchsize = Batchsize;
opts.silent = 1;
rfnn.learningRate = 1;
rfnn.scaling_learningRate = 0.99;

for iFnn=1:numel(rhiddenStruct)
    rfnn.W{iFnn}=sae.ae{end+1-iFnn}.W{2};
    rfnn.b{iFnn}=sae.ae{end+1-iFnn}.b{2};
end
rfnn=nntrain(rfnn,retrain_x,retrain_y,opts);
%}

%%%%%%%%%%%Ԥ��������ع��������%%%%%%%%%%%
train_x=train_x(1:2400,:);
train_y=train_y(1:2400,:);
trainingLength=2400;

dnn=nnff(dnn,train_x,train_y);
predict_y=dnn.a{end};
predict=train_y-predict_y;
save predict predict;


re_x=zeros(trainingLength,inputSize);
dnn=nnff(dnn,train_x,train_y);
rfnn=nnff(rfnn,dnn.a{end-1},re_x);
%fnn=nnff(fnn,train_x,zeros(trainingLength,hiddenStruct(end)));
%rfnn=nnff(rfnn,fnn.a{end},re_x);
re_x=rfnn.a{end};
reconstruction=train_x-re_x;
save reconstruction reconstruction;


thread=0;
letrainingLength=trainingLength;
letrain_x=reconstruction(:,6);
letrain_y=predict;


%{
letrainingLength=0;
letrain_x=[];
letrain_y=[];
for j=1:trainingLength
    if abs(reconstruction(j,6))>thread
        letrain_x=[letrain_x' train_x(j,6)']';
        letrain_y=[letrain_y' train_y(j,1)']';
        letrainingLength=letrainingLength+1;
    end
end
%}

%%%%%%%%%%%%%%%%%���Իع�%%%%%%%%%%%%%%%%%%%
letrain_x=[ones(letrainingLength,1) letrain_x];
theta=((letrain_x')*letrain_x)\(letrain_x')*letrain_y;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%test%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_x=zeros(testingLength,inputSize);
test_y=zeros(testingLength,outputSize);

for i=trainingLength+1:trainingLength+testingLength
    t=[];
    for j=1:length(inputPolluents)
        %ǰrange��Ŀ�������
        t=[t air_bj{station,1}(i:i+range-1,inputPolluents(j))'];
    end
    for j=1:length(weatherFactors)
        %��range�������Ԥ��
        %t=[t mean(air_bj{station,1}(i+range:i+range+timeDelay,weatherFactors(j))) max(air_bj{station,1}(i+range:i+range+timeDelay,weatherFactors(j))) air_bj{station,1}(i+range+timeDelay,weatherFactors(j))];
        t=[t air_bj{station,1}(i+range:i+range+timeDelay,weatherFactors(j))'];
    end
    test_x(i-trainingLength,:)=t;
    
    test_y(i-trainingLength,1)=air_bj{station,1}(i+range+timeDelay,outputPolluents);
    
end

dnn=nnff(dnn,test_x,test_y);
predict_y=dnn.a{end};
predict=test_y-predict_y;
disp(mean(abs(predict))*500);
%save predict predict;

re_x=zeros(testingLength,inputSize);
dnn=nnff(dnn,test_x,test_y);
rfnn=nnff(rfnn,dnn.a{end-1},re_x);
%fnn=nnff(fnn,test_x,zeros(testingLength,hiddenStruct(end)));
%rfnn=nnff(rfnn,fnn.a{end},re_x);
re_x=rfnn.a{end};
reconstruction=test_x-re_x;
%save reconstruction reconstruction;

%co=corrcoef((reconstruction(:,6)),(predict));
%disp(co(1,2));
scatter((reconstruction(:,6)),(predict(:,1)),'.','g');
hold on;

for j=1:testingLength
    if  abs(reconstruction(j,6))>thread
        predict_y(j)=predict_y(j)+[1 reconstruction(j,6)]*theta;
    end
end
predict=test_y-predict_y;
disp(mean(abs(predict))*500);

scatter((reconstruction(:,6)),(predict(:,1)),'.','r');
xlabel('reconstruction error');
ylabel('prediction error');


dnn=nntrain(dnn,train_x,train_y,opts);
dnn=nnff(dnn,test_x,test_y);
predict_y=dnn.a{end};
predict=test_y-predict_y;
disp(mean(abs(predict))*500);

end











