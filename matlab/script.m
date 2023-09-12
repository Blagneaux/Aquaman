clear;close all;
x = importdata("pressure08092023_x271-5y56.txt");
X = readmatrix("data_08092023_x271-5y56_exp1.xlsx");

t = x(1:269,1)*4.9/67.25;
s51 = x(1:269,2);

Time = X(:,1);
Donne = X(:,2);
ETAT = X(:,3);

index = find(ETAT);

%% Paramètre de l'aquisition reçu
fs = 1000;            % Sampling frequency                    
T = 1/fs;             % Sampling period       
L = numel(Donne);     % Length of signal

figure("Name","Simulation");
plot(t,s51);

xlabel("t");
ylabel("Pressure coefficient");

%% Supression de la composante continue (la moyenne)
Donne0 = Donne - mean(Donne);

%% FFT
Y = fft(Donne0);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;

%% Signal processing
%LP by BUTTER
fc1 = 40;
[b,a] = butter(2,fc1/(fs/2),'low');
filtrees=filtfilt(b,a,Donne0);

figure
plot(Time(index), -filtrees(index))
xlabel("t");
ylabel("Pressure");