testingImg = imresize(imread('TVL.jpg'),[255 255]);
referenceImg = imresize(imread('Trainscg1.jpg'),[255 255]);
% grayscale
testingGray = rgb2gray(testingImg);
referenceGray = rgb2gray(referenceImg);
% binarize
testingBw = imbinarize(testingGray) ;
referenceBw = imbinarize(referenceGray);
% compare *binary* images
[Fvalue,precision,recall,accuracy,JaccardIndex,TP,FP,TN,FN,FPrate,TPrate,MCC] = ...
    compareBinaryImages(referenceBw, testingBw);


I=imread('BO-TDyWT.jpg');
imshow(I);
A=mean2(I);
B=std2(I);
C=entropy(I);