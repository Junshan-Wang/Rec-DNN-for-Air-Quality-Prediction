function test_pr = knn(train_re, train_pr, test_re, k)
    dist=zeros(size(train_re,1));
    test_pr=[];
    m=size(train_re,1);
    for i=1:size(test_re,1)
        s=0;
        for j=1:m
            dist(j)=norm(train_re(j,:)-test_re(i,:));
        end 
        for j=1:k
            mini=10000;
            flag=1;
            for l=1:m
                if dist(l)<mini && dist(l)>=0
                    mini=dist(l);
                    flag=l;
                end
            end
            s=s+train_pr(flag);
            dist(flag)=-1;
        end
        s=s/k;
        test_pr=[test_pr' s]';
    end
end