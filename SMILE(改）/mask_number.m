function incompletely_labeled_matrix = mask_number(y,r)
% randomly mask m=1,2,3 labels of each labeled instance
% y:N*C label
% [n,~]=size(y);
% for i=1:n
%     su=sum(y(i,:));
%     t=0;
%     if su>m
%         sel=find(y(i,:)==1);
%         random=sel(randperm(numel(sel),m));
%         y(i,random)=0;
%     elseif su>1
%             t=su-1;
%             sel=find(y(i,:)==1);
%             random=sel(randperm(numel(sel),t));
%             y(i,random)=0;
%     end
% end
% incompletely_labeled_matrix=y;

real =find(y);
maskratios=fix(numel(real)*r);
random=real(randperm(numel(real),maskratios));
A(random)=0;
incompletely_labeled_matrix=y;
end
        
    