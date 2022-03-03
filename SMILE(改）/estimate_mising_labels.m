function estimate_matrix = estimate_mising_labels(y,L)
  %EXPANDY Summary of this function goes here
  %   Detailed explanation goes here
 [n,c]=size(y);%Y:N*C
 estimate_matrix=zeros(n,c);
 for i=1:n
     for j=1:c
         if y(i,j)==0
             estimate_matrix(i,j)=y(i,:)*L(:,j);
         else
             estimate_matrix(i,j)=1;
         end
     end
 end
 
 %normalize the data
  for i=1:n
%       aus=sum(estimate_matrix(i,:));
     aus=max(estimate_matrix(i,:));
     for j=1:c
         if y(i,j)==0&&aus~=0
             estimate_matrix(i,j)=estimate_matrix(i,j)/aus;
         end
     end
  end
             
             
 
             
             

