function sae = saesetup(size)
    for u = 2 : numel(size) % numel(����A)����������A�е�Ԫ�ظ���
        sae.ae{u-1} = nnsetup([size(u-1) size(u) size(u-1)]);% sizeΪ������Ԫ��������
    end
end
