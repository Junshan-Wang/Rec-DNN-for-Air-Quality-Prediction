function sae = saesetup(size)
    for u = 2 : numel(size) % numel(数组A)：返回数组A中的元素个数
        sae.ae{u-1} = nnsetup([size(u-1) size(u) size(u-1)]);% size为各层神经元数的数组
    end
end
