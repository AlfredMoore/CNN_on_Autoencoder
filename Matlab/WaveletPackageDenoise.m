function xd=WaveletPackageDenoise(nx,level)

[thr,sorh,keepapp,crit] = ddencmp('den','wp',nx);
xd = wpdencmp(nx,sorh,level,'db2',crit,thr,keepapp);

% [thr,sorh,keepapp] = ddencmp('den','wv',nx);
% xd_wv = wdencmp('gbl',nx,'db2',3,thr,sorh,keepapp);

% display(thr)
% display(sorh)
% display(keepapp)
% display(crit)


% clc,clear;
% load leleccum;
% load MotorLoad1HP_MotorSpeed1772rpm;
% nx = X097_DE_time;
% [thr,sorh,keepapp] = ddencmp('den','wv',nx);
% xd = wdencmp('gbl',nx,'db5',3,thr,sorh,keepapp);
% X097_DE_time_Denoised=xd;
% % subplot(211);
% % plot(nx);
% % title('含噪信号');
% % subplot(212);
% % plot(xd);
% % title('消噪后的信号');
% save('X097_DE_time_Denoised.mat','X097_DE_time_Denoised');