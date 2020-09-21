A = double(imread('bird_small.jpeg'));
function A_mod=distance(A, i, j)
  epsilon=10^-20;
  A_mod=(A(i+1, j, :)+ A(i-1, j, :)+ A(i, j+1, :)+ A(i, j-1, :))/4;
  if ((A(i, j, 1)-A(i+1, j, 1))^2+(A(i, j, 2)-A(i+1, j, 2))^2+(A(i, j, 3)-A(i+1, j, 3))^2) >epsilon
    if ((A(i, j, 1)-A(i-1, j, 1))^2+(A(i, j, 2)-A(i-1, j, 2))^2+(A(i, j, 3)-A(i-1, j, 3))^2) >epsilon
      if ((A(i, j, 1)-A(i, j+1, 1))^2+(A(i, j, 2)-A(i, j+1, 2))^2+(A(i, j, 3)-A(i, j+1, 3))^2) >epsilon
        if ((A(i, j, 1)-A(i, j-1, 1))^2+(A(i, j, 2)-A(i, j-1, 2))^2+(A(i, j, 3)-A(i, j-1, 3))^2) >epsilon
        return 
       endif
      endif
    endif
  endif
  
  A_mod=A(i, j, :);
  
endfunction


[m, n, ~]=size(A);
A=A/255;

R=A(:, :, 1);
B=A(:, :, 2);
G=A(:, :, 3);

grey1=R*0.2989+G*0.5870+B*0.1140;
subplot(1, 3, 1);
% B=imnoise(A, "gaussian");
imshow(grey1);
% for i=2:m-1
%   for j=2:n-1
%     A(i, j, :)=distance(A, i, j);
%   endfor
% endfor
% R=A(:, :, 1);
% B=A(:, :, 2);
% G=A(:, :, 3);

% grey2=R*0.2989+G*0.5870+B*0.1140;
%A=imsharpen(A);
%subplot(1, 2, 2);
%imshow(A);
size(grey1)
mea=sum(sum(grey1))/(m*n);
grey1=grey1-mea;
grey2=double(grey1>0.05);
subplot(1, 3, 2)
imshow(grey2);
grey3=double(grey1>0.07);
subplot(1, 3, 3)
imshow(grey3);