clear;clc;
% addpath('clf');
% addpath('lib');
% addpath('DataSets');
% addpath('basic classifier');
% addpath('MetricFunction');
% addpath(genpath('lib/manopt'));
% addpath('evl');

dataset='scene.txt';
data=importdata(dataset);
features = double(real(data(:,1:294)));%yeast=1:1000 m=1
labels=data(:,295:300);
param_features = features(:,:);
param_labels = labels(:,:);
res_features = features(:,:);
res_labels = labels(:,:);

% dataset_nums = [10];
% datasets = ["medical","arts","business","computers","education","entertainment","health","recreation","reference","science","social","society"];
% ins_nums = [978,7484,11184,12444,12030,12730,9205,12828,8027,6428,12111,14512];
% attri_nums = [1449,23146,21924,34096,27534,32001,30605,30324,39679,37187,52350,31802];
% lab_nums = [45,26,30,33,33,21,32,22,33,40,39,27];
% param_ins_nums = [978,300,300,300,300,300,300,300,300,300,300,300];
% param_attri_fres = [1,0.022,0.023,0.015,0.019,0.016,0.017,0.017,0.013,0.014,0.01,0.016];
% res_ins_nums = [978,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000];
% res_attri_fres = [1,0.1,0.1,0.07,0.09,0.07,0.08,0.08,0.06,0.06,0.05,0.08];
% lab_fres = [1,1,1,1,1,1,1,1,1,1,1,1];

% for i = dataset_nums
%     
% dataset=datasets(i) + ".txt";
% instance_num=ins_nums(i);
% attributes_num=attri_nums(i);
% label_num=lab_nums(i);
% label_frequent=lab_fres(i);

% %  计算最优参数
% param_attributes_frequent=param_attri_fres(i);
% [param_features,param_labels]=read_transform(dataset,instance_num,attributes_num,label_num,param_attributes_frequent,label_frequent);
% param_ins_num = param_ins_nums(i);
% param_features=param_features(1:param_ins_num,:);%%
% param_labels=param_labels(1:param_ins_num,:);%%
% fprintf('Start Run *** for dataset= %s time:%s \n',dataset,datestr(now));
alpha = 0;
beta = 0;
[~,best_alpha,best_beta] = ssmddm(param_features,param_labels,alpha,beta);

% % 计算最优解
% res_attributes_frequent=res_attri_fres(i);
% [res_features,res_labels]=read_transform(dataset,instance_num,attributes_num,label_num,res_attributes_frequent,label_frequent);
% res_ins_num = res_ins_nums(i);
% res_features=res_features(1:res_ins_num,:);%%
% res_labels=res_labels(1:res_ins_num,:);%%
[best_res,~] = ssmddm(res_features,res_labels,best_alpha,best_beta);

pathname = 'R:\comparing results\SSMDDM aut\';
% filename = char("res_" + datasets(i));
dataset_name = strsplit(dataset,'.');
filename = char("res_" + string(dataset_name(1)));
save([pathname,filename],'best_res');

% end