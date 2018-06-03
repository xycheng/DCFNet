% compute discrete FB bases on 3x3 and 5x5 patches

function [psi, c, kq_Psi]= calculate_FB_bases(L1)

% L1 = 1 (3x2) or 2 (5x5)

maxK = (2*L1+1)^2-1;

%% the grid

L = L1+1; %extended grid
R = L1+0.5;

truncate_freq_factor = 1.5;

if L1 < 2
    truncate_freq_factor = 2;
end

%%
[xx,yy] = meshgrid(-L1 : 1 : L1 , -L1 : 1 : L1);
xx=xx/(R);
yy=yy/(R);

ugrid1 = [xx(:), yy(:)];


%%
[xx,yy] = meshgrid(-L : 1 : L , -L : 1 : L);
xx=xx/(R);
yy=yy/(R);

ugrid = [xx(:), yy(:)];

%
[tgrid,rgrid] = cart2pol(ugrid(:,1),ugrid(:,2));
            % theta in (-pi, pi]
num_grid_points = size(ugrid,1);            

%% the FB basis

kmax = 15; 

load bessel.mat  %table for Bessel zeros that, [k,q, R_kq, R_{k,q+1}]     
B = bessel(bessel(:, 1) <=kmax & bessel(:, 4)<= pi*R*truncate_freq_factor, :);  
                        %Choose functions that k<=kmax, R_{k, q+1} \leq c*\pi N (N=R)
clear bessel



%%
[~, idxB] = sort( B(:,3), 'ascend'); %sorted by eigenvalue
mu_ns = B(idxB,3).^2;

ang_freqs = B(idxB, 1); %k
rad_freqs = B(idxB, 2); %q
R_ns = B(idxB, 3);

disp([ang_freqs, rad_freqs, mu_ns])

num_kq_all = numel(ang_freqs); 
max_ang_freqs = max(ang_freqs);

%%
Phi_ns=zeros(num_grid_points, num_kq_all); %amplitude of psi_kq (denoted as phi_ns)

% store into cell
Psi = [];
kq_Psi = [];
num_bases=0;

for i=1:size(B, 1)
    
    ki = ang_freqs(i);
    qi = rad_freqs(i);
    rkqi = R_ns(i);
    
    r0grid=rgrid*R_ns(i);
    
    [ F ]=besselj(ki, r0grid); %Bessel radial functions
                               %F(r) = J_k(r*R_kq) = J_kq(r)
                               %\int_0^1 J_kq(r) Phi_kq(r) r dr = 1/2*J_{k+1}(R_kq)^2
    
    Phi = 1./abs(besselj(ki+1, R_ns(i)))*F;
    Phi( rgrid >=1 ) = 0;
    
    
    Phi_ns(:, i)=Phi;
    
    %
    if ki ==0
        Psi = [Psi, Phi];
        kq_Psi = [kq_Psi, [ki,qi,rkqi]'];
        num_bases= num_bases+1;
    else
        Psi = [Psi, Phi.*cos(ki*tgrid)*sqrt(2), Phi.*sin(ki*tgrid)*sqrt(2)];
        kq_Psi = [kq_Psi, [ki,qi,rkqi]', [ki,qi,rkqi]'];
        num_bases= num_bases+2;
    end
    
    
end


num_bases = size(Psi,2);

%%
if num_bases > maxK
    Psi = Psi(:,1:maxK);
    kq_Psi = kq_Psi(:,1:maxK);
end
num_bases = size(Psi,2);

%% vis

% 
% figure(10),clf
% 
% for i=1: min(25,num_bases)
%     subplot(5,5,i);
%   
%     
%     scatter(ugrid(:,1), ugrid(:,2), 100, Psi(:,i), 's','filled' );
%     %scatter(ugrid(:,1), ugrid(:,2), 100, Phi(:,i), 's','filled' );
%     colormap(jet); %colorbar();
%     axis off
%     title(sprintf('k=%d, q=%d', kq_Psi(1,i), kq_Psi(2,i)))
% end
% %



%%
p = reshape(Psi, 2*L+1, 2*L+1, num_bases);


psi = p(2:end-1, 2:end-1,:);

psi = reshape( psi, (2*L1+1)^2, num_bases);



%% normalize
c = sqrt(mean(sum(psi.^2,1))); %c is a global absolute constant
psi = psi/c;

return;




return;

