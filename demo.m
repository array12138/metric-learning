clear all;close all;clc;
load('newTwomoons.mat');
beta = 1; mu = 1; 
[M_matrix,U_matrix]  = VirtualLMNN_1(newdata,newlabel,beta,mu);
showtime();