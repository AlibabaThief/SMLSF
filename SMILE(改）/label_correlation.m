function L = label_correlation(y,s)
%CORRELATION Summary of this function goes here
%   Detailed explanation goes here
% 该函数计算标记的相关性
%  y:label matrix: N*C标记矩阵
%  s:smoothness parameter 
%  返回标记间的相关性 L
% ----------------------------------------------%
[n,c]=size(y);
L=zeros(c,c);
for i=1:c
    for j=1:c
        index=0;
        for k=1:n
            if y(k,i)==1&&y(k,j)==1
                index=index+1;
            end
        end
        L(i,j)=(index+s)/(sum(y(:,i))+2*s);%why use sum(y(:,i)  
    end
end
        
                
    
