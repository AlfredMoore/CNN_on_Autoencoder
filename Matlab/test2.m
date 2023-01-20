maindir = 'E:\课程及其实验\毕业设计\DataFiles\IMS\4th_test\txt';
subdir = dir(maindir);
WP_level=4;
save_n = 1;
IG_a = [];
IG_b = [];
IG_m = [];

for i = 1:size(subdir)
    if( isequal(subdir(i).name , '.') || isequal( subdir(i).name, '..'))
        continue;
    end
    
    i
        
    data = importdata(fullfile(maindir,subdir(i).name));
    [K,n] = size(data);
    temp_data=zeros(K,n);
    sorted_data=zeros(K,n);
    for j = 1:n
        temp_data(:,j) = WaveletPackageDenoise(data(:,j),WP_level);
        sorted_data(:,j) = sort(temp_data(:,j));
    end
    
    IG_m_temp = median(sorted_data);
    M = fix(K/2);
    IG_a_temp = 2*sum(sorted_data(1:M))/M-IG_m_temp;
    IG_b_temp = 2*sum(sorted_data(M+1:K))/(K-M)-IG_m_temp;
    IG_a=[IG_a;IG_a_temp];
    IG_b=[IG_b;IG_b_temp];
    IG_m=[IG_m;IG_m_temp];
    
    

    
end