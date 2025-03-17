clear; close all;

% Loading data from text and Excel files
x = importdata("pressure08092023_x271-5y56_vx89_16.txt");
X = readmatrix("data_08092023_x242y56_exp1.xlsx");

% Processing time and pressure coefficient data
t = x(1:269,1)*4.9/67.25;
s51 = x(1:269,2);
s51 = s51 * 0.5 * 1000 * 0.089 * 0.089;

% Extracting columns from Excel data
Time = X(:,1);
Donne = X(:,2);
ETAT = X(:,3);

% Finding indices for non-zero ETAT values
index = find(ETAT);

% Acquisition parameters
fs = 1000;            % Sampling frequency                    
T = 1/fs;             % Sampling period       
L = numel(Donne);     % Length of signal

% Removing the DC component (mean)
Donne0 = Donne - mean(Donne);

% Fourier Transform and its processing
Y = fft(Donne0);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;

% Low pass filtering using Butterworth filter
fc1 = 9;
[b,a] = butter(2,fc1/(fs/2),'low');
filtrees = filtfilt(b,a,Donne0);

[d,c] = butter(2, fc1/(50), 'low');
s51_filtree = filtfilt(d,c,s51);

% Creating a combined figure for both signals
figure("Name","Combined Signal Plot");
plot(t, s51_filtree, 'b', 'DisplayName', 'Simulated Pressure');
hold on;
plot((Time(index)-min(Time(index)))/1000, -filtrees(index), 'r', 'DisplayName', 'Experimental Pressure');
xlabel("Time (s)");
ylabel("Pressure (Pa)");
legend show;
