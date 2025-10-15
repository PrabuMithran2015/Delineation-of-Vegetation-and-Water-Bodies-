noisyRGB=imread('CMA.jpg');
noisyRGB=imresize(noisyRGB,[225 225]);
figure,imshow(noisyRGB)
noisyR = noisyRGB(:,:,1);
noisyG = noisyRGB(:,:,2);
noisyB = noisyRGB(:,:,3);
net = denoisingNetwork('dncnn');
denoisedR = denoiseImage(noisyR,net);
denoisedG = denoiseImage(noisyG,net);
denoisedB = denoiseImage(noisyB,net);
denoisedRGB = cat(3,denoisedR,denoisedG,denoisedB);
figure,imshow(denoisedRGB)
title('Denoised Image')
imwrite(denoisedRGB, '123.jpg')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
noisyRGB=imread('123.jpg');
noisyRGB=imresize(noisyRGB,[225 225]);
noisyR = noisyRGB(:,:,1);
noisyG = noisyRGB(:,:,2);
noisyB = noisyRGB(:,:,3);
y=double(reshape(noisyR,[1,50625]));
t=double(reshape(noisyB,[1,50625]));
p = con2seq(y);
t = con2seq(t);
lrn_net = layrecnet(1,8);
lrn_net.trainFcn = 'trainrp';
lrn_net.trainParam.show = 3;
lrn_net.trainParam.epochs = 50;
lrn_net = train(lrn_net,p,t);
%lrn_net = train(lrn_net,p);
y = lrn_net(p);
figure,plot(cell2mat(y))
v=(cell2mat(y));
h=reshape(v,[225,225]);
figure,imagesc(h)
%figure,imshow(h)