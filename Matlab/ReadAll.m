
clc,clear;

maindir = 'E:\课程及其实验\毕业设计\DataFiles\IMS\2nd_test';
subdir = dir(maindir);
data_s = [];
for i = 1:size(subdir)
    if( isequal(subdir(i).name , '.') || isequal( subdir(i).name, '..'))
        continue;
    end
    data = importdata(fullfile(maindir,subdir(i).name));
    data_s = [data_s; data];
    i
end