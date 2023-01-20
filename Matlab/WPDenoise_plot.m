clc,clear;

maindir = 'E:\课程及其实验\毕业设计\DataFiles\IMS\2nd_test';
subdir = dir(maindir);
data = importdata(fullfile(maindir,subdir(3).name));
wave=data(:,1);

level = 3;
x_input=wave;
fs = 20000;
N = length(x_input);
[K,n] = size(data);

WP_level = level;
nx = x_input;
[thr,sorh,keepapp,crit] = ddencmp('den','wp',nx);
xd = wpdencmp(nx,sorh,level,'db2',crit,thr,keepapp);
for j = 1:n
	temp_data(:,j) = WaveletPackageDenoise(data(:,j),WP_level);
    sorted_data(:,j) = sort(temp_data(:,j));
end
IG_m_temp = median(sorted_data);
M = fix(K/2);
IG_a_temp = 2*sum(sorted_data(1:M))/M-IG_m_temp;
IG_b_temp = 2*sum(sorted_data(M+1:K))/(K-M)-IG_m_temp;

[thr,sorh,keepapp,crit] = ddencmp('den','wp',nx);
xd_level4 = wpdencmp(nx,sorh,4,'db2',crit,thr,keepapp);

[thr,sorh,keepapp,crit] = ddencmp('den','wp',nx);
xd_db5 = wpdencmp(nx,sorh,level,'db5',crit,thr,keepapp);

[thr,sorh,keepapp] = ddencmp('den','wv',nx);
xd_wv = wdencmp('gbl',nx,'db2',3,thr,sorh,keepapp);


figure(1)
plot(nx)
hold on 
plot(xd,"yellow")

figure(2)
subplot(2,2,1)
plot(xd)
title("三层小波包变换 db2小波基")
subplot(2,2,2)
plot(xd_level4)
title("四层小波包变换 db2小波基")
subplot(2,2,3)
plot(xd_db5)
title("三层小波包变换 db5小波基")
subplot(2,2,4)
plot(xd_wv)
title("三层小波变换")