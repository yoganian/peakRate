function [env, peakRate, peakEnv] = find_peakRate(sound, soundfs, onsOff, envtype)
% function [env, peakRate, peakEnv] = find_peakRate(sound, soundfs, onsOff, envtype)
% Inputs: 
%   sound - time x 1, sound waveform
%   soundfs - 1x1, sampling frequency of sound
%   onsOff  - 1 x 2 times of stimulus onset and offset in sound
%   envtype: 'loudness' (default), or 'broadband': specific loudness envelope or broadband envelope
% Output: 
%   env - amplitude envelope of input
%   peakRate - discrete time series of peakRate events in envelope
%   peakEnv - discrete time series of peakEnv events

% this function creates the speech envelope and a timeseries of discrete
% peakRate events, as used in Oganian & Chang, 2019, A speech envelope
% landmark for syllable encoding in human superior temporal gyrus, Science
% Advances, xxxx

% (c) Yulia Oganian, Oct 2019
% yulia.oganian@ucsf.edu

%% initialize
% need to know when signal contains speech to remode landmark events
% outside speech. this is t
if nargin<3
    onsOff= [1/soundfs length(sound)/soundfs];
end

if nargin<4, envtype = 'loudness'; end

% if set to 1, landmark series will be cleaned up to contain only a single
% peakRate event in each envelope cycle, defined as envelope
% trough-to-trough
cleanup_flag = 0;
%% get envelope
envfs=100;
switch envtype
    case 'loudness' %% specific loudness
        env = Syl_1984_Schotola(sound, soundfs);        
    case 'broadband' %% broadband envelope        
        rectsound  = abs(sound);
        [b,a] = butter(2, 10/(soundfs/2));
        cenv = filtfilt(b, a, rectsound);
        downsenv = resample(cenv, (1:length(cenv))/soundfs, 100);
        downsenv(downsenv <0) =0;
        env = downsenv;
end
%% get landmarks in envelope
%
% vector marking speech onset and offset times.
onsOffenv = zeros(length(env),2);
onsOffenv(1,ceil(onsOff(1)*envfs))=1;
onsOffenv(2,round(onsOff(2)*envfs))=1;


allTS = find_landmarks(env,  onsOffenv,cleanup_flag); 
peakEnv = allTS(4,:);
peakRate = allTS(6,:);

end

%% specific loudness function for envelope extraction
function Nm = Syl_1984_Schotola(p,fs)
%
%This identifies vowell nuclei according to the method of Schotola (1984).
%The calculation of modifed total loudness is identical to that of
%Syl_1979_Zwicker_etal_v1.m (e.g., Zwicker et al. 1979), but some
%additional rules/improvements are made here. This is continuation of work
%by Ruske and Schotola (1978, 1981), but this is the best-specified.
%
%INPUTS:
%p [N x 1]: speech waveform, p(t)
%fs [num]: sample rate (Hz)
%
%OUTPUTS:
%Nm [N x 1]: modified total loudness, Nm(t), used to identify VN and B


% written by Eric Edwards 
% adopted by Yulia Oganian, yulia.oganian@ucsf.edu

p = p(:);
N = length(p);
tN = (0:N-1)'./fs;
T = 1/fs;

%Loudness functions will be sampled at 100 Hz
sr = 100;
N1 = fix(N*sr/fs);
t1 = (0:N1-1)'./sr;

%Critical-band filters are applied in freq-domain
p = fft(p,N);
frqs = (0:N/2)*(fs/N); %FFT positive freqs
nfrqs = length(frqs);

%Set up for cricial-band filter bank (in freq-domain)
z = 13*atan(.76*frqs/1000) + 3.5*atan(frqs/7500).^2; %Bark (Zwicker & Terhardt 1980)
%CB=25+75*(1+1.4*(frqs/1000).^2).^.69; %Critical bandwidth (Zwicker & Terhardt 1980)
z = z(2:nfrqs-1); %to set DC and Nyquist to 0

%Set up for simple (RC) smoothing with 1.3-ms time-constant
tau = 0.0013; r = exp(-T/tau);
b1 = 1-r; a1 = poly(r);

F = zeros([N 1],'double'); %will hold critical-band filter shape
czs = 1:22; nczs = length(czs);
Nv = zeros ([N1 nczs],'double'); %will hold the specific loudnesses
for ncz=1:nczs, cz = czs(ncz);
    F(2:nfrqs-1) = 10.^(.7-.75*((z-cz)-.215)-1.75*(0.196+((z-cz)-.215).^2));
    %F = F./sum(F);
    Lev = real(ifft(p.*F,N));
    Lev = Lev.^2; %square-law rectification (Vogel 1975)
    Lev = filter(b1,a1,Lev); %smoothing (1.3-ms time-constant)
    Lev = flipud(filter(b1,a1,flipud(Lev)));
    %the last line makes zero-phase smoothing, comment out to leave causal
    
    Lev = log(Lev); %logarithmic nonlinearity; this is now "excitation level"
    %The "specific loudness", Nv(t), is made by "delog" of .5*Lev(t)
    Nv(:,ncz) = interp1q(tN,exp(.5*Lev),t1);
end
Nv = max(0,Nv); %in case negative values, set to 0

%Nm is the total modified loudness, Nm(t). This is LPFed by a 3-point
%triangular filter, n=26 times (I do 13 fwd and 13 backwd for zero-phase),
%which results in ~Gaussian smoothing with sig~=100 ms.
gv = ones([nczs 1],'double'); %weights
gv(czs<3) = 0; gv(czs>19) = -1;
Nm = Nv*gv;
b = ones([3 1],'double')./3; n = 13;
for nn = 1:n, Nm = filtfilt(b,1,Nm); end



%REFERENCES:
%
%Bismarck Gv (1973). Vorschlag f�r ein einfaches Verfahren zur
%Klassifikation station�rer Sprachschalle. Acustica 28(3): 186-7.
%
%Zwicker E, Terhardt E, Paulus E (1979). Automatic speech recognition using
%psychoacoustic models. J Acoust Soc Am 65(2): 487-98.
%
%Ruske G, Schotola T (1978). An approach to speech recognition using
%syllabic decision units. ICASSP: IEEE. 3: 722-5.
%
%Ruske G, Schotola T (1981). The efficiency of demisyllable segmentation in
%the recognition of spoken words. ICASSP: IEEE. 6: 971-4
%
%Schotola T (1984). On the use of demisyllables in automatic word
%recognition. Speech Commun 3(1): 63-87.

end


%% landmark detection in envelope
function [allTS, varNames] = find_landmarks(TS,  onsOff, cleanup_flag)

%% find discrete events in envelope

% make sure envelope is row vector
if ~isrow(TS)
    TS = TS';
end

% normalize envelope to between -1 and 1
TS = TS/max(abs(TS));

TS(find(onsOff(2,:)==1):end) = 0;
TS(1:find(onsOff(1,:)==1)) = 0;

% first temporal derivative of TS
diff_loudness = [diff(TS) 0];

%% discrete loudness
% min
[lmin, minloc] = findpeaks(-TS);
minEnv = zeros(size(TS));
minEnv(minloc)=-lmin;
% max
[lmin, minloc] = findpeaks(TS);
peakEnv = zeros(size(TS));
peakEnv(minloc)=lmin;

%% discrete delta loudness
% min
negloud = diff_loudness; negloud(negloud>0) = 0;
[lmin, minloc] = findpeaks(-negloud);
minRate = zeros(size(TS));
minRate(minloc)=-lmin;
% max
posloud = diff_loudness; posloud(posloud<0) = 0;
[lmin, minloc] = findpeaks(posloud);
peakRate = zeros(size(TS));
peakRate(minloc)=lmin;

clear negloud posloud ;
%% complete loudness information
allTS = [TS; ...
    diff_loudness;...
    minEnv;...
    peakEnv;...
    minRate;...
    peakRate];

%% --------------- clean up

if cleanup_flag
    %% start with maxima in envelope
    
    cmaxloc = find(allTS(4,:)~=0);
    cmax = allTS(4,cmaxloc);
    
    % initialize all other landmark variables
    cmin=nan(size(cmaxloc));cminloc=nan(size(cmaxloc));
    cminDt = nan(size(cmaxloc));cminDtLoc=nan(size(cmaxloc));
    cmaxDt=nan(size(cmaxloc));cmaxDtLoc = nan(size(cmaxloc)); 
    
    % --- define minima in envelope for each peak in envelope
    
    % first peak - getting cmin, cminloc
    cminloc(1) = 1;
    cmin(1) = 0.001;
        
    % remaining peaks
    for i= 2:length(cmaxloc)
        % find troughs between two consecutive peaks
        cExtrLoc = find(allTS(3,cmaxloc(i-1):cmaxloc(i))~=0);
        cExtr = allTS(3, cmaxloc(i-1)+cExtrLoc-1);
        if length(cExtr)==1 % this is standard case - one min per peak
            cmin(i) = cExtr;
            cminloc(i) = cmaxloc(i-1)+cExtrLoc-1;
        elseif length(cExtr) > 1 % if multiple troughs, use the lowest one; should not happen ever.
            [cmin(i),cl] = min(cExtr);
            cminloc(i) = cExtrLoc(cl)+cmaxloc(i-1)-1;
        elseif isempty(cExtr) % no minima in this window found by general algorithm; define as lowest point between this and previous peak.
            [cExtr(i), cExtrLoc(i)] = min(allTS(1, cmaxloc(i-1):cmaxloc(i)));
            cminloc(i) = cExtrLoc(i) + cmaxloc(i-1)+1;
            cmin(i) = cExtr(i);            
        end
    end
    
    %% % %  peakRate % % %
    
    
    for i= 1:length(cmaxloc)        
        if i == 1 % first peak
            cExtrLoc = find(allTS(6,1:cmaxloc(1))~=0);
            cExtr = allTS(6, cExtrLoc);
            prevloc = 0;
        else % remaining peaks
            cExtrLoc = find(allTS(6,cmaxloc(i-1):cmaxloc(i))~=0);
            cExtr = allTS(6,  cmaxloc(i-1)+cExtrLoc-1);
            prevloc = cmaxloc(i-1)-1;
        end
        
        if length(cExtr)==1
            cmaxDt(i) = cExtr;
            cmaxDtLoc(i) = cExtrLoc+prevloc;        
        elseif length(cExtr)>1
            [cmaxDt(i),cl] = max(cExtr);
            cmaxDtLoc(i) = cExtrLoc(cl)+prevloc;
        elseif isempty(cExtr)
            warning('no peakRate found in cycle %d  \n', i);
        end
    end
    
    %% % %  minRate % % %
    
    % all but last peaks
    for i= 1:length(cmaxloc)-1        
        cExtrLoc = find(allTS(5,cmaxloc(i):cmaxloc(i+1))~=0);
        cExtr = allTS(5, cmaxloc(i)+cExtrLoc-1);
        if length(cExtr)==1
            cminDt(i) = cExtr;
            cminDtLoc(i) = cExtrLoc+cmaxloc(i)-1;
        elseif isempty(cExtr)
            warning('no rate trough in cycle %d \n', i);
        elseif length(cExtr)>1
            [cminDt(i),cl] = min(cExtr);
            cminDtLoc(i) = cExtrLoc(cl)+cmaxloc(i)-1;        
        end
    end
    
    % last peak
    peakId = length(cmaxloc);
    envelopeEnd = find(TS~=0, 1, 'last');
    cExtrLoc = find(allTS(5,cmaxloc(end): envelopeEnd-1)~=0);
    cExtr = allTS(5, cExtrLoc+cmaxloc(end)-1);
    if length(cExtr)==1
        cminDt(peakId) = cExtr;
        cminDtLoc(peakId) = cExtrLoc+cmaxloc(end)-1;    
    elseif length(cExtr)>1
        [cminDt(peakId),cl] = min(cExtr);
        cminDtLoc(peakId) = cExtrLoc(cl)+cmaxloc(end)-1;
    elseif isempty(cExtr)
        warning('no minDtL in cycle %d \n', i);
    end
    
    %% detect silence in middle of utterance 
    if sum(cmin==0)>0 ,         warning('0 min found \n');    end
    
    %% combine all info
    cextVal = [cmin;cmax;cminDt;cmaxDt];
    cextLoc = [cminloc;cmaxloc;cminDtLoc;cmaxDtLoc];
    
    % redo allTS with cleaned values.
    for i = 3:6
        allTS(i,:)=0;
        allTS(i, cextLoc(i-2,:)) = cextVal(i-2,:);
    end
end
varNames = {'Loudness', 'dtLoudness', 'minenv', 'peakEnv', 'minRate', 'peakRate'};
end
