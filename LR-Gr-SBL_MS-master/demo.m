%%LR-SBL-CE
% Make performance comparision with GAMP

clc;
clear;
close all;
% rng(6)  %8,9

%% Simulation parameter settings
m = 16;                                   % Number of received antennas
n = 16;                                    % Number of transmit antennas
t = 15;                                     % Number of pilots
Num_Path = 3;
SNR = 20;
Bit = 3;                                    % Bit-depth of the Quantizer

%% Algorithm parameter settings
Num_Iter = 50;                     % Number of iterations



%% Baisc definition of the function
U_rx = dftmtx(m)/sqrt(m);      % DFT matrix
U_tx = dftmtx(n)/sqrt(n);       
Array_fun = @(Nvar,thetavar) exp(1j*(0:1:Nvar-1)'*thetavar');
%% Signal model generation

% Generate H

Path_gain = nan(Num_Path,1);
P = 1;                                     % The whole strength of the power
if Num_Path==1
    Path_gain = sqrt(P);
else
    Path_gain(1)   = sqrt(0.45*P)+sqrt(0.5*P*0.1)*randn(1,1);
    Path_gain(2:end) = sqrt(0.45*P/(Num_Path-1+eps))+sqrt(0.1*0.5*P/(Num_Path-1+eps))*randn(Num_Path-1,1);                        
end

Path_phase = 2* pi * (rand(Num_Path,1) - 0.5);
Path_amp = Path_gain.*exp(1i*Path_phase); 
omega = -pi+2*pi*rand(Num_Path,1);    % AoAs and AoDs
omega = sort(omega,'ascend');
phi = -pi+2*pi*rand(Num_Path,1);
% omega
% phi
H = zeros(m,n);
for path_index = 1:Num_Path
    H = H + Path_amp(path_index)*Array_fun(m,omega(path_index))*Array_fun(n,phi(path_index))';
end

X = (sign(randn(t,n))+1j*sign(randn(t,n)))/sqrt(2);
Z = H*(X.');
sigma = 10^(-SNR/10)*(norm(Z,'fro')^2/m/t);
Y = Z + sqrt(sigma/2)*(randn(size(Z)) +1j*randn(size(Z)));
% 10^(SNR/10)*sigma/P
if Bit==1
        h_norm_est = sqrt(max(norm(Y,'fro')^2-t*m*sigma,1)/t);       % Estimate the norm under one-bit quantization
else
        h_norm_est = [];
end

%% Quantization
y = Y(:);
Real_Imag_y = [real(y);imag(y)];
Quantizer.Bit = Bit;

sigma_z = P;
if Bit<inf    
    nbins = 2^Bit;
%     Delta_max = max(Real_Imag_y);
    Delta_max = 3*sqrt(n*sigma_z/2);
    Delta_min = -Delta_max;
    Stepsize = (Delta_max - Delta_min)/(nbins);
    Real_Imag_y_bin = floor((Real_Imag_y-Delta_min)/Stepsize);
    index_max = find(Real_Imag_y>=Delta_max);
    Real_Imag_y_bin(index_max) = nbins-1;
    index_min = find(Real_Imag_y<Delta_min);
    Real_Imag_y_bin(index_min) = 0;
else
    Delta_min = [];      % No quantization, and parameters are unneeded
    Stepsize = [];
    Real_Imag_y_bin = [];
end
Quantizer.Delta_min = Delta_min;
Quantizer.Stepsize = Stepsize;

[NMSE_MUSIC,NMSE] = TwostageLRSBLCE(Y, X, H, h_norm_est, Real_Imag_y_bin, Num_Iter, sigma, Quantizer);


%% Plot the results
Iter_index = 1:Num_Iter;
figure(1)
plot(Iter_index,NMSE,'-ro')
hold on
plot(Iter_index,NMSE_MUSIC*ones(Num_Iter,1),'-b+')
legend(sprintf('%d bit LR-Gr-SBL-CE',Bit ),sprintf('%d bit LR-SBL-CE+MUSIC',Bit))
ylabel('NMSE(dB)')
% legend(sprintf('%d bit LR-Gr-SBL-CE',Bit ),sprintf('%d bit GAMP',Bit ))
xlabel('Number of iterations')

