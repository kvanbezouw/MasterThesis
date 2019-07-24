clear all
close all



colour_sky = [55,94,151] ;
colour_sunset= [251,101,66];
colour_sunflower = [255,187,0];
colour_grass = [63,104,28];
colour_dark_navy = [1, 26, 39];
colour_string = [colour_sky; colour_sunset; colour_sunflower; colour_grass; colour_dark_navy] /255;
fontsize_overlay = 15;
fontsize_ingraph = 14;
fontsize_legend = 13;
scatter_size = 100; 

set(0,'defaulttextinterpreter','latex'); 
set(0, 'defaultAxesTickLabelInterpreter','latex'); 
set(0, 'defaultLegendInterpreter','latex')
set(0,'defaultFigureColor', [1 1 1]);
set(0,'defaultAxesFontSize',fontsize_overlay);


%(https://nl.mathworks.com/help/matlab/ref/imread.html)
to_transpose_or_not_to_transpose = 0; %edge finding ranks indices as if scanning from top to down. If you have a circle pointing to left or right, this fucks up indices (to y values belong to the same x-value, changing the order of the indices). 
material = "GaAs no IBE";
size = 6; %um. CHANGING THIS IS NEEDED TO CONVERT ALL ANGLES TO ACTUAL SURFACE LENGTHS!!
cut_off = 78; %fit gaussian around 2* cut_off points
disksize = strcat(num2str(size), " \mum");
[X, map]  = imread('C:\Users\Kees\Documents\MSc Scriptie\SEM\IBE\3_6um.tif');
decayrate = 0.001; 
edge_removal_pixel_cutoff = 200;
disk_blackwhite = imbinarize(X); %make black and white image
disk_blackwhite_cut = disk_blackwhite(1:(end - edge_removal_pixel_cutoff ), :); %Get rid of that bar below
if to_transpose_or_not_to_transpose == 1 %Not needed anymore because of improved sorting algorithm. Not in the mood to change this today
    disk_blackwhite_cut = transpose(disk_blackwhite_cut);
end

Kaverage = filter2(fspecial('average',3),disk_blackwhite_cut); %filters with a 3x3 gaussian (removes noise, connects edgepoints)
filtered_disk = imbinarize(Kaverage);

edg = edge(filtered_disk); % https://nl.mathworks.com/help/images/ref/edge.html. Edge has different methods for finding edges, default seems to work best.
% edg = edge(disk_blackwhite_cut);
contour = edg; %make copy so that edg stays untouched and can be used for vis. 
%filt_edg =  filter2(fspecial('average',3),edg); %filter again. Don't do
CC = bwconncomp(contour); %finds all connected lines
numPixels = cellfun(@numel,CC.PixelIdxList); %finds the number of pixels of each connected line, lines are indexed by arb. number
[largest,idx] = max(numPixels); %find largest line along with its id
number_of_contours = length(numPixels);
for i = 1:number_of_contours %set all lines that are not the largest to zero ( = black)
    if i ~= idx
        contour(CC.PixelIdxList{i}) = 0; 
    end
end
[r_x, r_y] = find(contour == 1);
r_x(:) = - r_x(:); %correct for te fact that direction of image is inverted

r_coord = zeros(max(length(r_x),length(r_y)),2);
r_coord(1:length(r_y),1) = r_x;
r_coord(1:length(r_x),2) = r_y;
circle_params = CircleFitByPratt(r_coord); %circle funct. defined below
%give circle params noicer names
x_center = circle_params(1);
y_center = circle_params(2);
radius = circle_params(3);

%circle the center around x,y = 0 for better visualisations and easier
%calulations
r_coord(:,1) = r_coord(:,1) - x_center;
r_coord(:,2) = r_coord(:,2) - y_center;
%find dr(theta)
%theta = atan2 %arctan always has caveats depending on sign of {x,y}. atan2 copes with these. theta = atan2(y,x)
theta = atan2(r_coord(:,2),r_coord(:,1));
theta_unwrapped = unwrap(theta); %unwrap to ensure no breaks in phaseplot @ \theta = +- k * 2 \pi)
[theta_ordered, sorted_theta_vec_indices] = sort(theta_unwrapped);

scaling = size/radius; %convert from number of pixels to a size (radius si given in pixels). Edgily assume that disk = exactly x um
circumference_len = theta_ordered* radius * scaling;
circumference_len = abs(circumference_len - circumference_len(1));
% dr = (sqrt(r_coord(:,1).^2 + r_coord(:,2).^2) - radius) .* scaling .* 1000; %[nm]
% for j = 1:length(r_coord)
%     dr(j) = (sqrt(r_coord(sorted_theta_vec_indices(j),1).^2 + r_coord(sorted_theta_vec_indices(j),2).^2) - radius)  .* scaling .* 1000;
% end
dr = (sqrt(r_coord(sorted_theta_vec_indices,1).^2 + r_coord(sorted_theta_vec_indices,2).^2) - radius).* scaling .* 1000;

dr_acf = acf(dr, length(dr)); %autocorrel. function defined below

%Use these coordinates to plot the fitted circle with (0,0) as center point
midpoint = round(length(circumference_len)/2);

circumference_len_left = fliplr(circumference_len(1:midpoint-1)); %flip from left to right to have the acf walk to the left effectively. Revert this later
circumference_len_right = circumference_len(midpoint:end);
dr_left = dr(1:midpoint-1);
dr_right = dr(midpoint:end);
dr_left_acf = acf(dr_left, length(dr_left));
dr_right_acf = acf(dr_right, length(dr_right));
dr = [fliplr(dr_left); dr_right];
dr_acf = [transpose(fliplr(dr_left_acf)); transpose(dr_right_acf)];
dr_acf = transpose(dr_acf);
%Find correlation length in the unchanged thingy
indices_gauss_data = round(midpoint - length(dr_acf)/cut_off : midpoint+length(dr_acf)/cut_off); %Taking first x datapoints seems to work better than the thing above

offset = (dr(max(indices_gauss_data))+ dr(min(indices_gauss_data)))/2;

x_gauss_data_total = circumference_len(indices_gauss_data);
y_gauss_data_total = dr_acf(indices_gauss_data) - offset;
gauss_and_offset = 'a.*exp(-((x-b)./c)^2) + d';
gauss_tot_x0 = [max(y_gauss_data_total), (x_gauss_data_total(end) + x_gauss_data_total(1))/2, (x_gauss_data_total(end) - x_gauss_data_total(1)) , 0];
gauss_fit_total = fit(x_gauss_data_total, y_gauss_data_total' ,gauss_and_offset,'Start', gauss_tot_x0);
gauss_coeffs_total = coeffvalues(gauss_fit_total);
gauss_coeffs_total = [gauss_coeffs_total(1:3) 0];
gauss_offset_funct = @(x, xdat) x(1) .* exp(-((xdat-x(2))./x(3)).^2) + x(4);

y_dat_fitted = gauss_offset_funct(gauss_coeffs_total, circumference_len);

[waviness, residue, residue_acf, c_n, theta_n, period] = find_waviness(circumference_len, dr, decayrate); 

circumference_len_left = fliplr(circumference_len(1:midpoint-1)); %flip from left to right to have the acf walk to the left effectively. Revert this later
circumference_len_right = circumference_len(midpoint:end);
residue_left = residue(1:midpoint-1);
residue_right = residue(midpoint:end);
residue_left_acf = acf(residue_left, length(residue_left));
residue_right_acf = acf(residue_right, length(residue_right));
residue = [fliplr(residue_left); residue_right];
residue_acf = [transpose(fliplr(residue_left_acf)); transpose(residue_right_acf)];
residue_acf = transpose(residue_acf);
[r_x_c, r_y_c] = circle(0, 0, radius, 0, 2*pi);

residue_tot = [residue fliplr(residue)];
residue_acf_tot= acf(residue, length(residue));
residue_acf_tot_2 = [fliplr(residue_acf_tot) residue_acf_tot];
%find correlation length (fit gauss over the gaussian regime of autocorr.
%== first few dat points)
%function
offset = mean(residue_acf);
residue_acf(:) = residue_acf(:) - offset; %duct-tape solution to avoide the need to fit both an offset and a gaussian to dataset. Add to auto corr later and fit
%indices_gauss_data = find((residue_acf./max(residue_acf)) > 0.3);
indices_gauss_data = round(midpoint - length(residue_acf)/cut_off : midpoint+length(residue_acf)/cut_off); %Taking first x datapoints seems to work better than the thing above
y_gauss_data_residue = residue_acf(indices_gauss_data);
x_gauss_data_residue = circumference_len(indices_gauss_data);
gauss_fit_residue = fit(x_gauss_data_residue, transpose(y_gauss_data_residue), 'gauss1');
gauss_coeffs_residue = coeffvalues(gauss_fit_residue);
dr_var = var(residue);



residue_tot = [residue fliplr(residue)];
residue_acf_tot= acf(residue, length(residue));
residue_acf_tot_2 = [fliplr(residue_acf_tot) residue_acf_tot];
cut_off_2 = 70;
offset_tot = mean(residue_acf_tot);
residue_acf_tot(:) = residue_acf_tot(:) - offset_tot; %duct-tape solution to avoide the need to fit both an offset and a gaussian to dataset. Add to auto corr later and fit
midpoint_tot = round(length(residue_acf_tot/2));
%indices_gauss_data = find((residue_acf./max(residue_acf)) > 0.3);
indices_gauss_data_tot = round(midpoint_tot - length(residue_acf_tot)/cut_off_2 : midpoint_tot + length(residue_acf_tot)/cut_off_2); %Taking first x datapoints seems to work better than the thing above
y_gauss_data_residue_tot = residue_acf_tot_2(indices_gauss_data_tot);
x_gauss_data_residue_tot = circumference_len(indices_gauss_data_tot - 3000) - mean(circumference_len(indices_gauss_data_tot - 3000));
gauss_fit_residue_tot = fit(x_gauss_data_residue_tot*1000, transpose(y_gauss_data_residue_tot), 'gauss1');
gauss_coeffs_residue_tot = coeffvalues(gauss_fit_residue_tot);

disp(strcat("Correlation length L_c = ", num2str(abs(gauss_coeffs_residue(3)*10^3)), "nm")) %Confirm that L_c is actually sqrt(sigma) later on
disp(strcat("Amplitude of roughness = ", num2str(sqrt(abs(gauss_coeffs_residue(1)))), "nm")) %Same here.  L_c looks correctish, amp doesnt. Fitting error bars are rather high
disp(strcat("Surface length =", num2str(max(circumference_len)), "\microm"))
%plot results
n = 3;
m = 3;

croppedImage = X(200:1750, 600:2370); % X((a,b), (c,d)), b = y from down
figure(3)

imshow(croppedImage);
figure(4)
disk_blackwhite_cropped = disk_blackwhite(200:1750, 600:2370);
imshow(disk_blackwhite_cropped)


figure(5)

plot( -theta_ordered,dr, 'Color', colour_string(1,:), 'LineWidth', 1.5)
ax = gca;
set(gca, 'Xtick', [0, pi/2, pi, 3*pi/2, 2*pi], ...
         'XTickLabel', ["0", "$\pi$/2", "$\pi$", "3$\pi$/2", "2 $\pi$"]);
xlabel('$\theta$')
ylabel('$\delta$r (nm)')


figure(24)
hold all
plot(r_coord(sorted_theta_vec_indices,2).*scaling,r_coord(sorted_theta_vec_indices,1).*scaling, 'Color', colour_string(1,:), 'LineWidth', 1.5)
plot(r_x_c.*scaling, r_y_c.*scaling, 'Color', colour_string(2,:), 'LineWidth', 1.5)


xlabel('x ($\mu$m)')
ylabel('y ($\mu$m)')
legend('Data',"Fit")
ax = gca;
ax.XLim = [2.3 5.5];
ax.YLim = [2.3 5.5];
   
   


figure(6)
plot( -theta_ordered, residue, 'color', colour_string(1,:), 'LineWidth', 1.5)
xlabel('$\theta $')
ylabel('$\delta$r (nm)')
set(gca, 'Xtick', [0, pi/2, pi, 3*pi/2, 2*pi], ...
         'XTickLabel', ["0", "$\pi$/2", "$\pi$", "3$\pi$/2", "2 $\pi$"]);
figure(8)
hold all
plot(x_gauss_data_residue_tot*1000, y_gauss_data_residue_tot, 'color',  colour_string(1,:), 'LineWidth', 1.5, 'DisplayName', 'Data')
plot(gauss_fit_residue_tot)
erbar_hdl_amp = errorbar(0, gauss_coeffs_residue_tot(1)/2, gauss_coeffs_residue_tot(1)/2)
erbar_hdl_amp = colour_string(3,:);
erbar_hdl_cor_len = errorbar(gauss_coeffs_residue_tot(3)/2 + 3, 15, gauss_coeffs_residue_tot(3)/2, 'horizontal')
erbar_hdl_amp = colour_string(4,:);
line([15 200], [25 25], 'Color', colour_string(5,:))
text( 200 + 15, 25, "$\sigma_{r}$", 'FontSize', fontsize_ingraph)

line([gauss_coeffs_residue_tot(3)/2 + 3 100], [14 8], 'Color', colour_string(5,:)) 
text( 100 + 15, 8, "L$_{c}$", 'FontSize', fontsize_ingraph)
xlabel('Surface length (nm)')
ylabel('Autocorrelation (nm$^2$)')
legend({'Data','Fit'})
% legend(strcat("L_c = ", num2str(abs(gauss_coeffs_residue(3)*10^3)), "nm\newline", "\sigma = ",num2str(sqrt(gauss_coeffs_residue(1))), "nm"))
%params = [x_cent, y_cent, radius] 




function [r_x, r_y] = circle(x,y,r,theta_min, theta_max)
hold on
th = [theta_min:(theta_max-theta_min)/2000:theta_max];
r_x = r * cos(th) + x;
r_y = r * sin(th) + y;
end

%Create function to seperate roughness (residual) from waviness, maxing to
%waviness of period 2pi/8. (fourierx doesnt go further and goign further
%doesnt seem to be needed)
function [waviness, residue, acf_residue, c_n, theta_n, period] = find_waviness(x, y, decayrate)
    decay = zeros(8, 1);
    n = 0;
    acf_residue = acf(y, length(y)); 
    waviness = cfit(); %assign location before filling it, otherwise waviness is destroyed after the while lopo ends #justmatlabthings
    decay(1) = abs(mean(acf_residue(round(end-end*0.4))/acf_residue(1)));
    while decay(n+1) >= decayrate && n < 8   %continue until adding sinusoids has no real effect anymore
       n = n+1;
       nth_fourier = strcat('fourier', num2str(n));
       waviness = fit(x, y, nth_fourier);
       residue = y - waviness(x);

       acf_residue = acf(residue, length(residue));

       decay(n+1) = abs(mean(acf_residue(round(end-end*0.2))/acf_residue(1))); %+1 because matlab indexing
    end
    if n > 0
        fourier_coeffs = coeffvalues(waviness);
        c_n = fourier_coeffs(1);
        theta_n(1) = 0;
        for j = 2:n
            a_n = j;
            b_n = j+2;
            c_n(j) = sqrt(fourier_coeffs(a_n) + fourier_coeffs(b_n));
            theta_n(j) = atan2(-fourier_coeffs(j+2), fourier_coeffs(j));
        end
        period = 2 * pi/ waviness.w;
        
    else
     residue = y;
     acf_residue = acf(residue, length(residue));
     c_n = 0;
     theta_n = 0;
     period = 0;
     disp("Signal already pretty residual (non-periodic). If unsatisfied, try raising decayrate")
    end
end



%create autocorrelation function, https://www.alanzucconi.com/2016/06/06/autocorrelation-function/
function autocorr = acf(y, t) %t = til which point to evaluate. commonly just take length(y) = ( = max(size(y)))
    y_mean = mean(y);
    N = length(y);

    % Numerator, unscaled covariance
    for i = 1:t
       cross_sum = zeros(N-t,1);
       for j = (i+1):N
           cross_sum(j) = (y(j) - y_mean).*(y(j-i) - y_mean);
       end
       %y_var = var(y);

       
       autocorr(i) = sum(cross_sum) / (N); 
    end
end


function Par = CircleFitByPratt(XY)
%--------------------------------------------------------------------------
%  
%     Circle fit by Pratt
%      V. Pratt, "Direct least-squares fitting of algebraic surfaces",
%      Computer Graphics, Vol. 21, pages 145-152 (1987)
%
%     Input:  XY(n,2) is the array of coordinates of n points x(i)=XY(i,1), y(i)=XY(i,2)
%
%     Output: Par = [a b R] is the fitting circle:
%                           center (a,b) and radius R
%
%     Note: this fit does not use built-in matrix functions (except "mean"),
%           so it can be easily programmed in any programming language
%
%--------------------------------------------------------------------------
n = size(XY,1);      % number of data points
centroid = mean(XY);   % the centroid of the data set
%     computing moments (note: all moments will be normed, i.e. divided by n)
Mxx=0; Myy=0; Mxy=0; Mxz=0; Myz=0; Mzz=0;
for i=1:n
    Xi = XY(i,1) - centroid(1);  %  centering data
    Yi = XY(i,2) - centroid(2);  %  centering data
    Zi = Xi*Xi + Yi*Yi;
    Mxy = Mxy + Xi*Yi;
    Mxx = Mxx + Xi*Xi;
    Myy = Myy + Yi*Yi;
    Mxz = Mxz + Xi*Zi;
    Myz = Myz + Yi*Zi;
    Mzz = Mzz + Zi*Zi;
end
   
Mxx = Mxx/n;
Myy = Myy/n;
Mxy = Mxy/n;
Mxz = Mxz/n;
Myz = Myz/n;
Mzz = Mzz/n;
%    computing the coefficients of the characteristic polynomial
Mz = Mxx + Myy;
Cov_xy = Mxx*Myy - Mxy*Mxy;
Mxz2 = Mxz*Mxz;
Myz2 = Myz*Myz;
A2 = 4*Cov_xy - 3*Mz*Mz - Mzz;
A1 = Mzz*Mz + 4*Cov_xy*Mz - Mxz2 - Myz2 - Mz*Mz*Mz;
A0 = Mxz2*Myy + Myz2*Mxx - Mzz*Cov_xy - 2*Mxz*Myz*Mxy + Mz*Mz*Cov_xy;
A22 = A2 + A2;
epsilon=1e-12; 
ynew=1e+20;
IterMax=20;
xnew = 0;
%    Newton's method starting at x=0
for iter=1:IterMax
    yold = ynew;
    ynew = A0 + xnew*(A1 + xnew*(A2 + 4.*xnew*xnew));
    if (abs(ynew)>abs(yold))
        disp('Newton-Pratt goes wrong direction: |ynew| > |yold|');
        xnew = 0;
        break;
    end
    Dy = A1 + xnew*(A22 + 16*xnew*xnew);
    xold = xnew;
    xnew = xold - ynew/Dy;
    if (abs((xnew-xold)/xnew) < epsilon), break, end
    if (iter >= IterMax)
        disp('Newton-Pratt will not converge');
        xnew = 0;
    end
    if (xnew<0.)
        fprintf(1,'Newton-Pratt negative root:  x=%f\n',xnew);
        xnew = 0;
    end
end
%    computing the circle parameters
DET = xnew*xnew - xnew*Mz + Cov_xy;
Center = [Mxz*(Myy-xnew)-Myz*Mxy , Myz*(Mxx-xnew)-Mxz*Mxy]/DET/2;
Par = [Center+centroid , sqrt(Center*Center'+Mz+2*xnew)];
end    %    CircleFitByPratt
% [Gmag,Gdir] = imgradient(disk_blackwhite); 


function figure_commands()
    grid on;
   
    box on;
end
