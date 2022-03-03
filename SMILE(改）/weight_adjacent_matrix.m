function W = weight_adjacent_matrix(X,knum)
%Using the kNN algorithm we will use the clusters to get a weight matrix
%this function is to construct a graph by knn   
%knum:the paramenter of knn ,stands for the number of nerib
%X:N*D
[n,~]=size(X);
D=zeros(n,n);
W=eye(n);
for i=1:n
    for j=i+1:n
       D(i,j)=sqrt(sum((X(i,:)-X(j,:)).^2));
       D(j,i)=D(i,j);
    end
end

Neighbors=cell(n,1); %Neighbors{i,1} stores the Num neighbors of the ith training instance
for i=1:n
    [~,index]=sort(D(i,:));
    Neighbors{i,1}=index(1:knum);
end

for i=1:n
    W(i,Neighbors{i,1})=1;
end

for j=1:n
    for i=1:n
        if ismember(i,Neighbors{j,1})==1
            W(i,j)=1;
        end
    end
end

He=W;
for i=1:n
    for j=1:n
        if He(i,j)==1
            W(i,j)=exp(-sqrt(sum((X(i,:)-X(j,:)).^2))/(10^2));
        end
    end
end
            






