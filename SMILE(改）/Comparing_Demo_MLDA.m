% clear;
% clc;

% dataset='scene.txt';
% fprintf('Start Run MLDA for dataset= %s time:%s \n',dataset,datestr(now));
% 
% data=importdata(dataset);
% data=a;
% features = double(real(data(:,1:1449)));
% labels=data(:,1450:1494);

datasets = ["medical","delicious","arts","business","computers","education","entertainment","health","recreation","reference","science","social","society"];
ins_nums = [978,16105,7484,11184,12444,12030,12730,9205,12828,8027,6428,12111,14512];
attri_nums = [1449,500,23146,21924,34096,27534,32001,30605,30324,39679,37187,52350,31802];
lab_nums = [45,983,26,30,33,33,21,32,22,33,40,39,27];
attri_fres = [1,1,0.1,0.1,0.07,0.09,0.07,0.08,0.08,0.06,0.06,0.05,0.08];
lab_fres = [1,0.3,1,1,1,1,1,1,1,1,1,1,1];
ins_re_nums = [978,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000];

for i = 1:size(datasets,2)

dataset=datasets(i) + ".txt";
instance_num=ins_nums(i);
attributes_num=attri_nums(i);
label_num=lab_nums(i);
attributes_frequent=attri_fres(i);
label_frequent=lab_fres(i);
[features,labels,attributes_Num,labels_Num]=read_transform(dataset,instance_num,attributes_num,label_num,attributes_frequent,label_frequent);
fprintf('Start Run *** for dataset= %s time:%s \n',dataset,datestr(now));
features=features(1:ins_re_nums(i),:);
labels=labels(1:ins_re_nums(i),:);

labels=labels';

if(min(min(labels))<=-1)    % labels c*N
    labels(labels<0)=0;
end
      
minT=10;% the minimum size of member proteins
fun_stat=sum(labels,2);
sel_fun_idx=find(fun_stat>=minT);
labels=labels(sel_fun_idx,:);% 删除小于30的样本的标记
insl=sum(labels,1);
insl_zeros=find(insl==0);
insf=sum(features,2);
insf_zeros=find(insf==0);
ins_zeros=unique([insl_zeros insf_zeros']);
features(insl_zeros,:)=[];
labels(:,insl_zeros)=[];
fprintf(' delete instances %d\n',length(ins_zeros));
labels(labels==0)=-1;
[Nlabel,mNum]=size(labels);
labels=labels';

%parameters setting
reduced_dim=Nlabel-1;
clear opts;
opts.reg_eig =0.5;
opts.k=reduced_dim;

% times = 10;%10
% fold = 10;
% [num_sample, ~] = size(features);
% for itrator=1:times%times
%     indices = crossvalind('Kfold', num_sample, fold);
%     for rep=1:fold%fold
%         testIdx = find(indices == rep);
%         trainIdx = setdiff(find(indices),testIdx);
%         test_feature = features(testIdx,:);
%         test_labels = labels(testIdx,:);
%         train_feature = features(trainIdx,:);
%         train_labels = labels(trainIdx,:);

label_original=labels;

round = 10;
test_number=ceil(mNum*0.3);  % 测试集的大小
boundary=mNum-test_number;

for run=1:round
    fprintf('\n run %d time : %s\n',run,datestr(now));
    tic
    
    index=randperm(mNum);
    %incomplete_labelnum=ceil(boundary*Numratio);
    
    train_feature=features(index(1:boundary),:);
    train_labels=label_original(index(1:boundary),:);
    test_feature=features(index(boundary+1:boundary+test_number),:);
    test_labels=label_original(index(boundary+1:boundary+test_number),:);
    
    tic
    proj_vec =  MLDA(train_feature', train_labels', opts);
    train_feature_reduction = train_feature*proj_vec;
    test_feature_reduction = test_feature*proj_vec;
    fprintf('\n finish MLDA time= %s\n',datestr(now));
    
    % MLKNN
    % Training Phase
    KNN_Num = 10;%6
    Smooth = 1;%需要知道k近邻的紧邻数量和平滑参数
%         [train_data,~] = mapminmax(train_feature_reduction,0,1); %归一化处理
%         train_target = train_labels';
%         [Prior,PriorN,Cond,CondN,neighbor_labels,temp,temp_Ci] = train(train_data,train_target,KNN_Num,Smooth);
%         % Testing Phase
%         test_target = test_labels';
%         [test_data,~] = mapminmax(test_feature_reduction,0,1); %归一化处理
%         [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Macro_AvgF1,Micro_AvgF1,Outputs,Pre_Labels]...
%          = test(train_data,train_target,test_data,test_target,KNN_Num,Prior,PriorN,Cond,CondN,neighbor_labels,temp,temp_Ci);
%         %cd('./measures');
% %         mea(rep,1)=sorensendist(test_distribution, pre_distribution);
% %         mea(rep,2)=kldist(test_distribution, pre_distribution);
% %         mea(rep,3)=chebyshev(test_distribution, pre_distribution);
% %         mea(rep,4)=intersection(test_distribution, pre_distribution);
% %         mea(rep,5)=cosine(test_distribution, pre_distribution);
%         mea(rep,1)=HammingLoss;
%         mea(rep,2)=RankingLoss;
%         mea(rep,3)=OneError;
%         mea(rep,4)=Coverage;
%         mea(rep,5)=Average_Precision;
%         mea(rep,6)=Macro_AvgF1;
%         mea(rep,7)=Micro_AvgF1;

    % Begin MLKNN
    [Prior,PriorN,Cond,CondN]=MLKNN_train(train_feature_reduction,train_labels',KNN_Num,Smooth); % Invoking the training procedure
    [Output]=MLKNN_test(train_feature_reduction,train_labels',test_feature_reduction,test_labels',KNN_Num,Prior,PriorN,Cond,CondN);
    Pre_Labels = sign(Output);%添加MLKNN
    test_target = test_labels';
    
    HammingLoss=Hamming_loss(Pre_Labels,test_target);
    RankingLoss=Ranking_loss(Pre_Labels,test_target);
    OneError=One_error(Pre_Labels,test_target);
    Coverage=coverage(Pre_Labels,test_target);
    Average_Precision=Average_precision(Pre_Labels,test_target);
    Macro_AvgF1=Macro_Avg_F1(Pre_Labels,test_target);
    Micro_AvgF1=Micro_Avg_F1(Pre_Labels,test_target);
    
    mea(run,1) = Micro_AvgF1;
    mea(run,2)=Macro_AvgF1;
    mea(run,3)=1-RankingLoss;
    mea(run,4)=1-OneError;
    mea(run,5)=Average_Precision;
    mea(run,6)=1-HammingLoss;
    mea(run,7)=Coverage;

        %cd('../');
        fprintf('=========================== %d times %d cross ( %d seconds )======================= \n', run, round, toc);
        
    end
%     res_mean(itrator,:) = mean(mea,1);
%     res_std(itrator,:) = std(mea,1);
% end

res(1,:) = mean(mea, 1);
res(2,:) = std(mea, 1);

pathname = 'R:\comparing results\MLDA\';
filename1 = char("mea_" + datasets(i));
filename2 = char("res_" + datasets(i));
save([pathname,filename1],'mea');
save([pathname,filename2],'res');
end
