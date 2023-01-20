clc,clear
n=4;
load 2nd_test.mat
figure()
for i = 1:n
subplot(2,2,i)
length = size(IG_a,1);
k = 1;

IG_a1 = IG_b;
IG_b1 = IG_a;
scatter(1:k:length,IG_a1(1:k:length,i),'.')

hold on
scatter(1:k:length,IG_m(1:k:length,i),'.')

hold on
scatter(1:k:length,IG_b1(1:k:length,i),'.')
subtitle("Bearing"+i)
% legend('a','m','b')
end