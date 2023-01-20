
clc,clear;

maindir = 'E:\课程及其实验\毕业设计\DataFiles\IMS\2nd_test';
subdir = dir(maindir);
data = importdata(fullfile(maindir,subdir(3).name));
wave=data(:,1);

level = 3;
x_input=wave;
fs = 20000;
N = length(x_input);


wpt=wpdec(x_input,level,'dmey');        %进行3层小波包分解
plot(wpt);                          %绘制小波包树
wpviewcf(wpt,1);                    %时间频域图
nodes = [2^level-1:2^(level+1)-2]';
ord = wpfrqord(nodes);
nodes_ord = nodes(ord);
for i=1:2^level
rex3(:,i)=wprcoef(wpt,nodes_ord(i));  %实现对节点小波节点进行重构        
end
 
figure;                          %绘制第3层各个节点分别重构后信号的频谱
for i=0:2^level-1
subplot(4,2^level/4,i+1);
x_sign=rex3(:,i+1); 
N=length(x_sign); %采样点个数
signalFFT=abs(fft(x_sign,N));%真实的幅值
Y=2*signalFFT/N;
f=(0:N/2)*(fs/N);
plot(f,Y(1:N/2+1));
ylabel('amp'); xlabel('frequency');grid on
axis([0 inf 0 0.05]);
title(['小波包第',num2str(level),'层',num2str(i),'节点信号频谱']);
end

% E = wenergy(wpt);

% wavelet packet coefficients. 求取小波包分解的各个节点的小波包系数
cfs3_0=wpcoef(wpt,nodes_ord(1));  %对重排序后第3层0节点的小波包系数0-8Hz
cfs3_1=wpcoef(wpt,nodes_ord(2));  %对重排序后第3层0节点的小波包系数8-16Hz
cfs3_2=wpcoef(wpt,nodes_ord(3));  %对重排序后第3层0节点的小波包系数16-24Hz
cfs3_3=wpcoef(wpt,nodes_ord(4));  %对重排序后第3层0节点的小波包系数24-32Hz
cfs3_4=wpcoef(wpt,nodes_ord(5));  %对重排序后第3层0节点的小波包系数32-40Hz
cfs3_5=wpcoef(wpt,nodes_ord(6));  %对重排序后第3层0节点的小波包系数40-48Hz
cfs3_6=wpcoef(wpt,nodes_ord(7));  %对重排序后第3层0节点的小波包系数48-56Hz
cfs3_7=wpcoef(wpt,nodes_ord(8));  %对重排序后第3层0节点的小波包系数56-64Hz
 
E_cfs3_0=norm(cfs3_0,2)^2;  %% 1-范数：就是norm(...,1)，即各元素绝对值之和；2-范数：就是norm(...,2)，即各元素平方和开根号；
E_cfs3_1=norm(cfs3_1,2)^2;
E_cfs3_2=norm(cfs3_2,2)^2;
E_cfs3_3=norm(cfs3_3,2)^2;
E_cfs3_4=norm(cfs3_4,2)^2;
E_cfs3_5=norm(cfs3_5,2)^2;
E_cfs3_6=norm(cfs3_6,2)^2;
E_cfs3_7=norm(cfs3_7,2)^2;
E_total=E_cfs3_0+E_cfs3_1+E_cfs3_2+E_cfs3_3+E_cfs3_4+E_cfs3_5+E_cfs3_6+E_cfs3_7;
 
p_node(1)= 100*E_cfs3_0/E_total;           % 求得每个节点的占比
p_node(2)= 100*E_cfs3_1/E_total;           % 求得每个节点的占比
p_node(3)= 100*E_cfs3_2/E_total;           % 求得每个节点的占比
p_node(4)= 100*E_cfs3_3/E_total;           % 求得每个节点的占比
p_node(5)= 100*E_cfs3_4/E_total;           % 求得每个节点的占比
p_node(6)= 100*E_cfs3_5/E_total;           % 求得每个节点的占比
p_node(7)= 100*E_cfs3_6/E_total;           % 求得每个节点的占比
p_node(8)= 100*E_cfs3_7/E_total;           % 求得每个节点的占比
 
figure;
x=1:8;
bar(x,p_node);
title('各个频段能量所占的比例');
xlabel('频段');
ylabel('能量百分比/%');
for j=1:8
text(x(j),p_node(j),num2str(p_node(j),'%0.2f'),...
    'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
end