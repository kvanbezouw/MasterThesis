clear all
close all
set(0,'DefaultFigureWindowStyle','normal')
number_of_runs_to_fit = 18;%
fontsize_overlay = 15;
fontsize_ingraph = 13;
fontsize_legend = 10.8;
colour_sky = [55,94,151] ;
colour_sunset= [251,101,66];
colour_sunflower = [255,187,0];
colour_grass = [63,104,28];
colour_dark_navy = [1, 26, 39];
colour_string = [colour_sky; colour_sunset; colour_sunflower; colour_grass; colour_dark_navy] /255;

nr_rel = number_of_runs_to_fit;
% colour_least_power = colour_string(1,:);
% colour_max_power = colour_string(2,:);
colour_max_power = [255,80,10]/255;
colour_least_power = [41,80,188]/255;
colour_map = [linspace(colour_least_power(1), colour_max_power(1), nr_rel); linspace(colour_least_power(2), colour_max_power(2), nr_rel);  linspace(colour_least_power(3), colour_max_power(3), nr_rel) ];
line_width = 1;
marker_size = 10;
set(0,'defaulttextinterpreter','latex'); 
set(0, 'defaultAxesTickLabelInterpreter','latex'); 
set(0, 'defaultLegendInterpreter','latex')
set(0,'defaultFigureColor', [1 1 1]);
set(0,'defaultAxesFontSize',fontsize_overlay);

colour_resonance = colour_string(1,:);
LineWidth = 1.5;

number_of_runs = 20;
material = "GaAs";
disksize = "6 micron disk";
mode = "TE_{r=1}";
wl_num = 1527,1
wl = strcat(num2str(wl_num), "nm") ;
offset_toggle = 0; % 0 means take average. 1 means take leftmost dat point to normalise data , 2 means rightmostdatpoint
%titles =  strcat(material, ", ",disksize, ", ", mode ,", ", wl)
titles = "Test"
run = "1";
run_name = strcat("C:\Users\Kees\Documents\MSc Scriptie\PowerMeasAnalysis\Si\si_6micron_disk_powermeas_analysis\si_6micron_disk_powermeas_analysis\te_r1_1555,52nm_powermeas_run1_"); %include a _ at the end

fit_init = 1; %this number should be teh start of where to use gamma_nla for fitting (linaer regime)
excluded_wl_over_pd_dat_points = []; % First few datapoints can be sloppy because of the small wl cahnges involved 
% excluded_wl_over_pd_dat_points = [1 4 5 6 9 10 11 12];
%save_file_name = strcat(material,"_tm_r1_", num2str(wl), '_powermeas_run', num2str(run)); % include a _ at the end
save_file_name = "dummy"
% USE THIS FOR FITTING
min_datapoint = 001; % (min = 1 ) %points used for all images
max_datapoint = 9000; % (max = 9000)
min_fit_index = 0001;  %points used for fitting of cold cav only (MZ data vs normed transmission
max_fit_index = max_datapoint - min_datapoint -5000; 
C =  1.82e-6;
DoublePeak = 1;
cutoff = 0.85;
manual_doublet_x0 = 1; % 0 = not manual, 1 is manual estimation of linewidhts etc.
init_doublet_fitting_params = [-0.2, 800, 1000, -0.25, 800, -9000, 1]; % amp_left, lw_left, cen_left, amp_right, lw_right, cen_right, offset 
init_t_analytic_fitting_params = [50, -10000, 419.5, 423.5, 457.1]; %%a = kappa_e, b = center of initial resonance, c = k_b, d = k_tot_left, e = k_tot_right;
wavelength_correction = 1;
offset_normalisation = 1;
wavelength_correction_init_value = 1;
power_init_values = [1:3];
power_mismatch = 0;
slice = fit_init :number_of_runs_to_fit;
C_0 = C; %Initial fit param.
cold_cav_dataset = 1;
c = 3 * 10^8;
omega = 2 * pi * c/(wl_num * 10^-9);

for i = 0:number_of_runs
    imported_power_measurement(i+1) = importdata(strcat(run_name,num2str(i)));
end

n = 3; %grid for suplots
m = 3;
figure(300)

figure(4)
offset = zeros(number_of_runs,1);
for j = 0:number_of_runs
    all_data_run = imported_power_measurement(j+1).data;
    all_data_run_t = transpose(all_data_run);
    x_dat_temp = all_data_run_t(4,min_datapoint:max_datapoint);
    y_dat_temp = all_data_run_t(5,min_datapoint:max_datapoint);
    x_dat_temp = x_dat_temp(~isnan(x_dat_temp)); % fourier transforming etc. has padded the data slightley
    y_dat_temp = y_dat_temp(~isnan(y_dat_temp));
    x_dat_intermediate(j+1, :) = x_dat_temp(1:end);    %Select region where the magic happens
    y_dat(j+1, :) = y_dat_temp(1:end); 
    if offset_toggle == 0
        offset(j+1) = (mean(y_dat(j+1, 1:100)) + mean(y_dat(j+1, end-100:end)))/2;
    elseif offset_toggle == 1
        offset(j+1) = mean(y_dat(j+1, 1:100));
    elseif offset_toggle == 2
        offset(j+1) = mean(y_dat(j+1, end-100:end));
    elseif offset_toggle == 3
        if j < 17

            [min_offset_value, min_offset_index ]= min(y_dat(j+1,:));
            offset(j+1) = mean(y_dat(j+1,min_offset_index + 100:min_offset_index + 200));
        end
        if i >= 17
            offset(j+1) = mean(y_dat(j+1, end-100:end));
        end
    end
    %[y_minima(j+1), indices_y_minima(j+1)] = min(y_dat(j+1,:));
    %x_minima(j+1) = x_dat_intermediate(j+1,indices_y_minima(j+1));
    y_dat_normied_intermediate(j+1,:) = y_dat(j+1, :)/offset(j+1);
    subplot(3,7, j+1)
    plot(x_dat_intermediate(j+1,:), y_dat(j+1, :))
end



y_dat_normied_intermediate(end, :) = lowpass(y_dat_normied_intermediate(end, :), 0.0001);
for j = 0:(number_of_runs)
    x_dat(j+1, :) = x_dat_intermediate(j+1, 100:(end-100));
    y_dat_normied(j+1, :)  = y_dat_normied_intermediate(j+1, 100:(end-100)) ./ y_dat_normied_intermediate(end, 100:(end-100));
    [y_minima(j+1), indices_y_minima(j+1)] = min(y_dat_normied(j+1,:));
    x_minima(j+1) = x_dat(j+1,indices_y_minima(j+1));
end
minimum_coldcav = min(x_dat(1,:));
%Cold cavity data
x_dat_coldcav = x_dat(cold_cav_dataset , min_fit_index:max_fit_index);
y_dat_coldcav = y_dat_normied(cold_cav_dataset ,min_fit_index:max_fit_index);

%sgtitle(strcat(material, ", ",disksize, ", ", wl))

figure(301)
plot(x_dat_coldcav', y_dat_coldcav')

% legend('data', strcat('Fit. (Left lw = ', num2str(kappa_t_left_mhz(1),4),' \pm ', num2str(abs(left_lorentz_lw_bound_mhz),4), ' MHz, \kappa_i = ', num2str(kappa_t_left_mhz(1) - kappa_e_mhz,4), 'MHz.\newlineRight lw = ', num2str(kappa_t_right_mhz,4), ' \pm ', num2str(abs(right_lorentz_lw_bound_mhz),4),' MHz, \kappa_i = ', num2str(kappa_t_right_mhz(1) - kappa_e_mhz,4), 'MHz.\newline\gamma_\beta = ', num2str(kappa_b_mhz,4), 'MHz)'))

title(titles)

xlabel("Detuning (MHz)")
ylabel("Transmission (norm.)")



for j = 1:number_of_runs
    wavelengths_shifts_MHz(j) = x_minima(j) - x_minima(1);
    wavelengths_shifts_pm_uncorrected(j) = wavelengths_shifts_MHz(j) * 7.7 * 10^(-3); 
end

%Correct for laser drift
if wavelength_correction == 1
    for j = 1:(number_of_runs - 1)
        wavelengths_shifts_pm(j) = wavelengths_shifts_pm_uncorrected(j) - (wavelengths_shifts_pm_uncorrected(end)- wavelengths_shifts_pm_uncorrected(1))/(number_of_runs - 1)*(j-1);
    end
     %wavelengths_shifts_pm(:) = wavelengths_shifts_pm(:) + wavelengths_shifts_pm_uncorrected(fit_init)  - (wavelengths_shifts_pm_uncorrected(end)- wavelengths_shifts_pm_uncorrected(1))/(number_of_runs - 1)*(fit_init-1)
elseif wavelength_correction == 0
     wavelengths_shifts_pm = wavelengths_shifts_pm_uncorrected;
end



amp = all_data_run_t(8);
powers_temp = all_data_run_t(7,:);
amp = 100;
powers_approx = 3.3 * powers_temp(~isnan(powers_temp)); %max power is 3.3 mW, attenuations are calibrated to that value
%Normalise powers against photodiode power.
if offset_normalisation == 1
    powers = powers_approx(1:20) .* offset(1:20)'/(amp*powers_approx(1)) * 0.397/1.57 * 1.3  ; %divide through the first power (which is then one), multiply by offset to get baselinevoltage
elseif offset_normalisation == 0
    powers = powers_approx(1:20) .* offset(1)'/(amp*powers_approx(1)) * 0.397/1.57 * 1.3; %divide through the first power (which is then one), multiply by offset to get baselinevoltage
end
if power_mismatch == 1
    powers = powers_approx * offset(6)/offset(1);
end
P_d = powers .* (1-y_minima(1:end-1));
wavelengths_shifts_pm_pre = wavelengths_shifts_pm;
wl_offset = (wavelengths_shifts_pm_pre(max(power_init_values))- wavelengths_shifts_pm_pre(min(power_init_values)))/(P_d(max(power_init_values)) - P_d(min(power_init_values))) * P_d(min(power_init_values));
if wavelength_correction_init_value == 1
        wavelengths_shifts_pm = wavelengths_shifts_pm + wl_offset;
end

%Function declaration
T_analytic_t_norm = 'abs(-1 - a/2*(1/(-d/2+1i*((x-b) + c/2))+1/(-e/2+1i*((x-b) - c/2))))^2'; %a = kappa_e, b = center of initial resonance, c = k_b, d = k_tot_left, e = k_tot_right;
T_analytic_t_norm_singlet = 'abs(-1 - a/2*1/(-b/2+1i*(x-c)))^2';
    
if DoublePeak == 0
    
   [Lorentz_x0, T_analytic_x0] = find_initial_fitting_conditions_singlet(x_dat_coldcav, y_dat_coldcav);
   T_analytic_fit = fit(x_dat_coldcav', y_dat_coldcav', T_analytic_t_norm_singlet, 'Start', T_analytic_x0);
   T_analytic_x= coeffvalues(T_analytic_fit);
   T_analytic_x_confint = confint(T_analytic_fit, 0.68);
   [kappa_e_mhz, resonance_center_mhz, kappa_b_mhz, kappa_t_left_mhz, kappa_t_right_mhz] = deal(T_analytic_x(1),T_analytic_x(3), 0, 0, T_analytic_x(2));
   left_lorentz_lw_bound_mhz = 0;
   right_lorentz_lw_bound_mhz = abs(T_analytic_x_confint(1,2) - kappa_t_right_mhz);  
   kappa_i_left_mhz = 0;
   kappa_i_right_mhz = kappa_t_right_mhz - kappa_e_mhz;
   ki_hz = kappa_i_right_mhz(1) * 10^6;
   kappa_t_mhz = kappa_t_right_mhz(1);
   [gamma_nla_sim, K_sim, T_min_theory] = simulate_gamma_nla(y_minima(slice), kappa_e_mhz, kappa_b_mhz, kappa_i_right_mhz,100000 );

   
   
elseif DoublePeak == 1
    [Lorentz_x0, T_analytic_x0] = find_initial_fitting_conditions(x_dat_coldcav, y_dat_coldcav, init_doublet_fitting_params, init_t_analytic_fitting_params, manual_doublet_x0, cutoff);
    
    T_analytic_fit = fit(x_dat_coldcav', y_dat_coldcav', T_analytic_t_norm, 'Start', T_analytic_x0);
    T_analytic_x= coeffvalues(T_analytic_fit);
    T_analytic_x_confint = confint(T_analytic_fit, 0.68);
    T_analytic_coeffnames = coeffnames(T_analytic_fit);
    [kappa_e_mhz, resonance_center_mhz, kappa_b_mhz, kappa_t_left_mhz, kappa_t_right_mhz] = deal(T_analytic_x(1),T_analytic_x(2), T_analytic_x(3), T_analytic_x(4),T_analytic_x(5));
    left_lorentz_lw_bound_mhz = abs(T_analytic_x_confint(1,4) - kappa_t_left_mhz);
    right_lorentz_lw_bound_mhz = abs(T_analytic_x_confint(1,5) - kappa_t_right_mhz);
    kappa_b_mhz_bound = abs(T_analytic_x_confint(1,3) - kappa_b_mhz)
    for j = 0:(number_of_runs)
        x_dat(j+1, :) = x_dat(j+1,:) - resonance_center_mhz;
        x_minima(j+1) = x_dat(j+1,indices_y_minima(j+1));
    end
    resonance_center_mhz = 0;
    T_analytic_fit.(T_analytic_coeffnames{2}) = 0;
    %Cold cavity data
    x_dat_coldcav = x_dat(cold_cav_dataset , min_fit_index:max_fit_index);
    y_dat_coldcav = y_dat_normied(cold_cav_dataset ,min_fit_index:max_fit_index);



    
    [gamma_nla_sim, K_sim, T_min_theory] = simulate_gamma_nla(y_minima(slice), kappa_e_mhz, kappa_b_mhz, kappa_t_left_mhz - kappa_e_mhz, kappa_t_right_mhz - kappa_e_mhz);
    
    
    
    % If left peak is deeper
    if kappa_t_right_mhz > kappa_t_left_mhz
%         K_smalldip =  kappa_e_mhz/kappa_t_right_mhz;
        
        kappa_i_left_mhz = kappa_e_mhz  ./ K_sim;
        kappa_i_right_mhz = kappa_t_right_mhz - kappa_e_mhz;
        ki_hz = kappa_i_left_mhz(1) * 10^6;
        kappa_t_mhz = kappa_t_left_mhz(1);
    % If right peak is deeper
    elseif kappa_t_right_mhz < kappa_t_left_mhz
%         K_smalldip =  kappa_e_mhz/(kappa_t_left_mhz);
        kappa_i_left_mhz = kappa_t_left_mhz - kappa_e_mhz;
        kappa_i_right_mhz = kappa_e_mhz./ K_sim ;
        kappa_t_mhz = kappa_t_right_mhz(1);
        ki_hz = kappa_i_right_mhz(1) * 10^6;
    end
     
end
% U_c = [1./(kappa_e_mhz * 10^6) .* K_sim .* P_d(slice)] * 10^-3;
% U_c = P_d(slice)/(gamma_nla_sim + kappa_i_mhz)
k_t_mhz = kappa_e_mhz./K_sim + kappa_e_mhz;
% U_c = 1/(2*pi)*[(1./(kappa_e_mhz*10^6)) .* K_sim' .* P_d(slice)] * 10^-3;
U_c = 1/(2*pi)*[(1./(k_t_mhz'*10^6)).* P_d(slice)] * 10^-3;

% k_i_mhz = K_sim'/(kappa_e_mhz*10^6) + kappa_t_mhz;
%mind to use the new gamma_nla_dash for fitting gamma_la
gamma_nla_dash_sim_vs_pd_fit = fit(P_d(slice)', gamma_nla_sim,'poly1', 'Exclude' ,excluded_wl_over_pd_dat_points)
gamma_nla_dash_sim_vs_pd_dash_x= coeffvalues(gamma_nla_dash_sim_vs_pd_fit)/kappa_t_mhz;

gamma_nla_sim_vs_Uc_fit = fit(U_c' * 10^15,transpose(transpose(gamma_nla_sim)), 'poly1', 'Exclude' ,excluded_wl_over_pd_dat_points)
gamma_nla_sim_vs_Uc_x = coeffvalues(gamma_nla_sim_vs_Uc_fit);
gamma_nla_sim_vs_Uc_confint = confint(gamma_nla_sim_vs_Uc_fit, 0.68);
gamma_nla_sim_vs_Uc_bound = abs(gamma_nla_sim_vs_Uc_confint(1,1) - gamma_nla_sim_vs_Uc_x(1));
gamma_nla_vs_Uc_sim_fit_MHz_fJ_datpoints = gamma_nla_sim_vs_Uc_x(1).* U_c*10^15;


% gamma_nla_dash_sim_vs_pd_dash_x(1) = gamma_nla_dash_sim_vs_pd_dash_x(1)  * 4.5;
    

%Now use the gamma_nla_dash fit to extract gamma'_la and C
borselli_funct_lin = strcat('a*x*(b+',num2str(gamma_nla_dash_sim_vs_pd_dash_x(1)),'*x)/(1+',num2str(gamma_nla_dash_sim_vs_pd_dash_x(1)),'*x)'); %a = C, b = gamma_la, 
borselli_funct_lin_vs_pd = strcat('a*(b+',num2str(gamma_nla_dash_sim_vs_pd_dash_x(1)),'*x)/(1+',num2str(gamma_nla_dash_sim_vs_pd_dash_x(1)),'*x)'); %a = C, b = gamma_la, 
borselli_funct_lin_vs_pd_test = strcat('a*(b+c*x)/(1+c*x)');
x0_bors = [C_0, 0.1]; 
x0_bors_test = [1400/10, 0.1, 8];

bors_funct_fit_test =  fit(P_d(slice)',transpose(wavelengths_shifts_pm(slice)./P_d(slice)), borselli_funct_lin_vs_pd_test, 'Start', x0_bors_test, 'Exclude' ,excluded_wl_over_pd_dat_points);
x_bors_funct_test = coeffvalues(bors_funct_fit_test);


% Initially, wavelength shift wasn't taken as normalised against P_d.
bors_funct_fit_2 = fit(P_d(slice)',transpose(wavelengths_shifts_pm(slice)./P_d(slice)), borselli_funct_lin_vs_pd, 'Start', x0_bors, 'Exclude' ,excluded_wl_over_pd_dat_points);
x_bors_lin_values_2 = coeffvalues(bors_funct_fit_2);
x_bors_lin_confint_2 = confint(bors_funct_fit_2, 0.68);
C_fit_2 = x_bors_lin_values_2(1);
C_fit_bound_2 = abs(x_bors_lin_confint_2(1,1)-C_fit_2);
gamma_la_dash_fit_2  = x_bors_lin_values_2(2);
gamma_la_dash_fit_bound_2 = abs(x_bors_lin_confint_2(2,2)-gamma_la_dash_fit_2);
bors_fit_datpoints_2 = C_fit_2.*(gamma_la_dash_fit_2 + gamma_nla_dash_sim_vs_pd_dash_x(1).*P_d(slice))./(1 + gamma_nla_dash_sim_vs_pd_dash_x(1).*P_d(slice));



bors_funct_fit = fit(P_d(slice)',wavelengths_shifts_pm(slice)', borselli_funct_lin, 'Start', x0_bors);
x_bors_lin_values = coeffvalues(bors_funct_fit);
x_bors_lin_confint = confint(bors_funct_fit, 0.68);
C_fit = x_bors_lin_values(1);
C_fit_bound = abs(x_bors_lin_confint(1,1)-C_fit);
gamma_la_dash_fit = x_bors_lin_values(2);
gamma_la_dash_fit_bound = abs(x_bors_lin_confint(2,2)-gamma_la_dash_fit)
bors_fit_datpoints = C_fit.*P_d(slice).*(gamma_la_dash_fit + gamma_nla_dash_sim_vs_pd_dash_x(1).*P_d(slice))./(1 + gamma_nla_dash_sim_vs_pd_dash_x(1).*P_d(slice));


P_abs_W = (1/C) .* wavelengths_shifts_pm * 10^-12;
 %W/J = Hz
%c_fit = fit(transpose([0 U_c(fit_init+1:number_of_runs_to_fit+1)]),transpose([0 P_abs_W(fit_init+1:(number_of_runs_to_fit+1))]),'poly2'); %fit in W/J
c_fit = fit(transpose([0 transpose(U_c)']),transpose([0 transpose(P_abs_W(slice))']),'poly2'); %fit in W/J

PabsW_vs_Uc_fit= coeffvalues(c_fit); %W/J
PabsW_vs_Uc_confint = confint(c_fit, 0.68); %W/J
 
gamma_nla_fit_error = abs(PabsW_vs_Uc_confint(1,1) - PabsW_vs_Uc_fit(1)); %Hz/J
gamma_la_fit_error = abs(PabsW_vs_Uc_confint(1,2) - PabsW_vs_Uc_fit(2)); %Hz
absorption = PabsW_vs_Uc_fit(1).*U_c.^2 + PabsW_vs_Uc_fit(2).*U_c + PabsW_vs_Uc_fit(3); %W
absorption_mW = absorption * 10^3; %mW
absorption_linear = PabsW_vs_Uc_fit(2).*U_c + PabsW_vs_Uc_fit(3);
absorption_linear_mW = absorption_linear  * 10^3;





detuning = linspace(-kappa_b_mhz/2 - 100,kappa_b_mhz/2 + 100,3000);
for i = 1:length(detuning)
    T_init(i)  = get_T(detuning(i), kappa_e_mhz, kappa_b_mhz, kappa_i_left_mhz(1) + kappa_e_mhz, kappa_i_right_mhz(1) + kappa_e_mhz);
end
% 
% for i = 1:length(detuning)
%     T_init_2(i)  = get_T(detuning(i), kappa_e_mhz, kappa_b_mhz, kappa_i_left_mhz(1) - 2000 + kappa_e_mhz, kappa_i_right_mhz(1) - 2000);
% end
% 
ylim_min_trans = 0.76;
ylim_max_trans = 1.05;
line_width = 1;
marker_size = 60;
marker_type = '*'
%Normalised transmission spectra
h1 = figure(44)
figure_commands()
hold all
rel_runs = round(linspace(1,number_of_runs_to_fit,number_of_runs_to_fit/5));
rel_runs(2) = rel_runs(2)+2;
for i = 1:number_of_runs_to_fit
    if ismember(i,rel_runs)
        plot(x_dat(i, :)/1000, y_dat_normied(i, :),'Color',colour_map(:,i), 'LineWidth', line_width)
    end
    scatter(x_minima(i)/1000, y_minima(i),marker_size, colour_string(5,:), '*')
end
plot(x_dat(end-1, :)/1000, y_dat_normied(end-1, :),'Color',colour_string(5,:), 'LineWidth', line_width, 'LineStyle', '--')
scatter(x_minima(end-1)/1000, y_minima(end-1),marker_size, colour_string(5,:), '*')

xlabel("Detuning (GHz)")
ylabel("Transmission (a.u.)")
ax = gca;
ax.YLim = [ylim_min_trans ylim_max_trans];
ax.XLim = [min(x_dat(i,:))/1000, max(x_dat(1,:))/1000];

set(h1,'Position',[10 10 560*1.3 420])

T_analytic_x0 = T_analytic_x0/1000;
T_analytic_x0(2) = 0;
T_analytic_fit_2 = fit(x_dat_coldcav'./1000, y_dat_coldcav', T_analytic_t_norm, 'Start', T_analytic_x0);




fig_coldcav = figure(49)

hold all
figure_commands()
p2 = plot(x_dat_coldcav'./1000, y_dat_coldcav')
p1 = plot(T_analytic_fit_2)
set(p1, 'color', colour_string(2,:))
set(p2, 'color', colour_map(:,1))
v=0.2;
base = 0.85;
txt1 = text(2.2,base+0.2*v, strcat(' $\gamma_{s,i}/2\pi$ = ', num2str((kappa_t_left_mhz(1) - kappa_e_mhz)/1000,2), '$\pm$', num2str(abs(left_lorentz_lw_bound_mhz)/1000,2), ' GHz'), 'Fontsize', fontsize_ingraph)
txt2 = text(2.2,base+0.1*v, strcat(' $\gamma_{c,i}/2\pi$ = ', num2str((kappa_t_right_mhz(1) - kappa_e_mhz)/1000,2), '$\pm$', num2str(abs(right_lorentz_lw_bound_mhz)/1000,2), ' GHz'), 'Fontsize', fontsize_ingraph)
txt3 = text(2.2,base, strcat('$\gamma_\beta/2\pi$ =  ', num2str(kappa_b_mhz/1000,2),'$\pm$', num2str(kappa_b_mhz_bound/1000,2), ' GHz'), 'Fontsize', fontsize_ingraph)

ax = gca;
ax.YLim = [ylim_min_trans ylim_max_trans];
ax.XLim = [-8.5 8.5];
% legend('data', strcat('Fit. (Left $\gamma_t = $', num2str(kappa_t_left_mhz(1),4),' $\pm$ ', num2str(abs(left_lorentz_lw_bound_mhz),4), ' MHz, $\gamma_i$ = ', num2str(kappa_t_left_mhz(1) - kappa_e_mhz,4), 'MHz.  Right $\gamma_t = $', num2str(kappa_t_right_mhz,4), ' $\pm$ ', num2str(abs(right_lorentz_lw_bound_mhz),4),' MHz, $\gamma_i$ = ', num2str(kappa_t_right_mhz(1) - kappa_e_mhz,4), 'MHz.$\gamma_\beta$ = ', num2str(kappa_b_mhz,4), 'MHz)'))
legend('Data','Fit')
% text(-5000
xlabel("Detuning (GHz)")
ylabel("Transmission (a.u.)")
fig_po_dur_leg(fig_coldcav,"test_rob_pdf_save2")

wl_shift_sc_size = 80;
figure(50)
hold all
figure_commands()

sc1 = scatter(P_d(slice), wavelengths_shifts_pm(slice)./P_d(slice), wl_shift_sc_size, 'filled',marker_type)
sc2 = scatter(P_d(excluded_wl_over_pd_dat_points), wavelengths_shifts_pm(excluded_wl_over_pd_dat_points)./P_d(excluded_wl_over_pd_dat_points),wl_shift_sc_size, colour_string(5,:), 'filled', marker_type)

sc1.MarkerEdgeColor = colour_string(1,:);
gamma_la_dash_fit_bound_2 = sqrt(gamma_la_dash_fit_bound_2^2 + (gamma_nla_sim_vs_Uc_bound/gamma_nla_sim_vs_Uc_x(1))^2);
sc2.MarkerEdgeColor = colour_string(5,:);
plot(P_d(slice), bors_fit_datpoints_2,'Color', colour_string(2,:))
xlabel("$P_d$ (mW)")
ylabel("$\Delta \lambda / P_d$ (nm/mW)")
legend('Included data points', 'Fit', 'Location', 'NorthWest')
% legend('data', strcat('fit ( C = ', num2str(C_fit_2,2),"\pm ", num2str(C_fit_bound_2,2), "\newline\gamma_{la} = ", num2str(gamma_la_dash_fit_2*ki_hz*10^-6, 3), "\pm ", num2str(gamma_la_dash_fit_bound_2*ki_hz*10^-6, 2), "MHz)"))
text(0.025, 880, strcat('C = ', num2str(C_fit_2,2)," $\pm$ ", num2str(C_fit_bound_2,2), "pm/mW"), 'Fontsize', fontsize_ingraph)
text(0.025, 990, strcat("$\gamma_{\mathrm{la}}/2\pi$ = ", num2str(gamma_la_dash_fit_2*ki_hz*10^-6/1000, 2), "$\pm$", num2str(gamma_la_dash_fit_bound_2*ki_hz*10^-6/1000, 2), "GHz"), 'Fontsize', fontsize_ingraph)


figure(900)
figure_commands()
hold all
sc3 = scatter(U_c(slice)* 10^15,gamma_nla_sim', wl_shift_sc_size, 'filled',marker_type)
sc4 = scatter(U_c(excluded_wl_over_pd_dat_points)* 10^15,gamma_nla_sim(excluded_wl_over_pd_dat_points)',wl_shift_sc_size, colour_string(5,:), 'filled', marker_type)
sc3.MarkerEdgeColor = colour_string(1,:);
sc4.MarkerEdgeColor = colour_string(5,:);
plot(U_c*10^15, gamma_nla_vs_Uc_sim_fit_MHz_fJ_datpoints,'Color', colour_string(2,:))
ax = gca;
ax.YLim = [1.2*min(gamma_nla_sim') 1.2*max(gamma_nla_sim')];
text(11/2,26, strcat('$\partial\gamma_{\mathrm{TPA}}/(2\pi\partial U_c)$ = ', num2str(gamma_nla_sim_vs_Uc_x(1),2), "$\pm$",  num2str(gamma_nla_sim_vs_Uc_bound,2), "MHz/fJ"), 'Fontsize', fontsize_ingraph)
ylabel('$\gamma_{\mathrm{TPA}}/2\pi$ (MHz)')
xlabel('U$_c$(fJ)')
legend('Included data points', 'Linear part of fit', 'Location', 'NorthWest')
% legend('Data', strcat('fit (\partial\gamma_{nla}/\partialU_c = ', num2str(gamma_nla_sim_vs_Uc_x(1)), "\pm",  num2str(gamma_nla_sim_vs_Uc_bound), "MHz/fJ"))


fitUc_vs_wl = fit(U_c(slice)'.*10^15, wavelengths_shifts_pm(slice)', 'poly2');
Uc_vs_wl_coeffvalues = coeffvalues(fitUc_vs_wl);
Uc_vs_wl_lin = Uc_vs_wl_coeffvalues(2);
Uc_vs_wl_datpoints = Uc_vs_wl_lin.*U_c(slice).*10^15;
figure(9001)
figure_commands()
hold all
sc3 = scatter(U_c(slice)* 10^15,wavelengths_shifts_pm(slice), wl_shift_sc_size, 'filled',marker_type)
sc3.MarkerEdgeColor = colour_string(1,:);
sc4.MarkerEdgeColor = colour_string(5,:);
plot(U_c(slice).*10^15, Uc_vs_wl_datpoints, 'LineStyle', '--')

xlabel('$U_c$ (fJ)')
ylabel('$\Delta \lambda$ (pm)')
legend('Data', 'Linear absorption', 'Location', 'NorthWest')
% legend('Data', strcat('fit (\partial\gamma_{nla}/\partialU_c = ', num2str(gamma_nla_sim_vs_Uc_x(1)), "\pm",  num2str(gamma_nla_sim_vs_Uc_bound), "MHz/fJ"))
ax = gca;
ax.XLim = [0, 1.1*max(U_c(slice))*10^15];


% figure(300)
% subplot(n,m,6)
% scatter([0 U_c(2:end)] * 10^15, [0 gamma_nla_sim'].*ki_hz*10^-6)
% hold on
% plot([0 U_c(2:end)] * 10^15, gamma_nla_dash_fit_datpoints.*ki_hz*10^-6) 
% title(titles)
% ylabel('gamma_{nla} (MHz/fJ)')
% xlabel('U_c(fJ)')
% legend('Data', strcat('fit (\partial\gamma_{nla}/\partialU_c = ', num2str(gamma_nla_vs_Uc_dash_x(1)*ki_hz*10^-6*10^-15), "\pm", num2str(gamma_nla_vs_dash_Uc_bound*ki_hz*10^-6*10^-15,2), "MHz/fJ"))
% 
% figure(300)
% subplot(n,m,2)
% plot(T_analytic_fit,x_dat_coldcav', y_dat_coldcav')
% 
% legend('data', strcat('Fit. (Left \gamma_t = ', num2str(kappa_t_left_mhz(1),4),' \pm ', num2str(abs(left_lorentz_lw_bound_mhz),4), ' MHz, \gamma_i = ', num2str(kappa_t_left_mhz(1) - kappa_e_mhz,4), 'MHz.\newlineRight \gamma_t = ', num2str(kappa_t_right_mhz,4), ' \pm ', num2str(abs(right_lorentz_lw_bound_mhz),4),' MHz, \gamma_i = ', num2str(kappa_t_right_mhz(1) - kappa_e_mhz,4), 'MHz.\newline\gamma_\beta = ', num2str(kappa_b_mhz,4), 'MHz)'))
% 
% title(titles)
% 
% xlabel("Detuning (MHz)")
% ylabel("Transmission (norm.)")
% 
% figure(300)
% subplot(n,m,3)
%  
% yyaxis left
% plot(U_c .* 10^15, P_abs_W(slice)*10^3 , 'Marker', 'x','LineStyle','--') %mW
% hold on
% 
% %yyaxis right
% 
% plot(U_c*10^15,absorption_mW,'LineStyle','-') 
% hold on
% plot(U_c*10^15, absorption_linear_mW,'LineStyle','-') 
% title(titles)
% ylabel('Power (mW)')
% 
% fit_legend_text = strcat('Polyn. Fit: (\gamma_{la} = ', num2str(PabsW_vs_Uc_fit(2)*10^-6,3), ' \pm ', num2str(gamma_la_fit_error * 10^-6,2), 'MHz\newline\gamma_{nla} = ',num2str(PabsW_vs_Uc_fit(1)*10^(-6)*10^(-15),3), ' \pm ', num2str(gamma_nla_fit_error*10^(-6)*10^(-15),2), 'MHz/fJ)'); %c(2) was in W/fJ =Ts
% 
% 
% yyaxis right
% plot([0 U_c] .* 10^15, [0 P_d(slice)], 'Marker','x', 'LineStyle','--')
% hold on
% xlabel('Intracavity energy (fJ)')
% ylabel('Power (mW)')
% hold on
% legend('P_{abs}',strcat('P_{abs,fit}' , fit_legend_text), 'P_{la,fit}','P_d','Location', 'northwest')
%   
% figure(300)
% subplot(n,m,4)
% scatter(P_d(slice), wavelengths_shifts_pm(slice))
% hold on
% plot(P_d(slice), bors_fit_datpoints)
% title(titles)
% xlabel('P_{d} (mW)')
% ylabel('\Delta \lambda')
% % legend('bors lin fit', 'bors cub fit','data','quadratic fit','cubic fit')
% legend('data', strcat('fit ( C = ', num2str(C_fit,3),"\pm ", num2str(C_fit_bound,2), "\newline\gamma_{la} = ", num2str(gamma_la_dash_fit*ki_hz*10^-6, 3), "\pm ", num2str(gamma_la_dash_fit_bound*ki_hz*10^-6, 2), "MHz)"))
% 
% figure(300)
% subplot(n,m,5)
% scatter(linspace(1,number_of_runs + 1,number_of_runs + 1), y_dat(1:end,min_datapoint),'x')
% hold on
% scatter(linspace(1,number_of_runs + 1,number_of_runs + 1), y_dat(1:end,end),'o')
% hold on
% scatter(linspace(1,number_of_runs + 1,number_of_runs + 1), offset,'x')
% legend('First datapoints', 'Last datapoints', 'Offsets')
% title(titles)
% xlabel("Run numbers")
% ylabel("Transmissions")
% 
% figure(300)
% subplot(n,m,8)
% plot(powers_approx(1:10))
% hold on
% plot(powers(1:10))
% ylabel('power')
% legend('Approximate powers', 'Scaled powers')
% xlabel('data point')
% 
% 
% figure(300)
% subplot(n,m,7)
% scatter(P_d(slice), wavelengths_shifts_pm(slice)./P_d(slice))
% hold on
% plot(P_d(slice), bors_fit_datpoints_2)
% xlabel("P_d")
% ylabel("\Delta \lambda / P_d")
% legend('data','fit')
% legend('data', strcat('fit ( C = ', num2str(C_fit_2,3),"\pm ", num2str(C_fit_bound_2,2), "\newline\gamma_{la} = ", num2str(gamma_la_dash_fit_2*ki_hz*10^-6, 3), "\pm ", num2str(gamma_la_dash_fit_bound_2*ki_hz*10^-6, 2), "MHz)"))
% 
% 
% title(titles)
% subplot(n,m,6)
% 
% scatter([0 U_c] * 10^15, [0 gamma_nla_sim'] )
% hold on
% plot(U_c*10^15, gamma_nla_vs_Uc_sim_fit_MHz_fJ_datpoints)
% title(titles)
% ylabel('\gamma_{nla} (MHz)')
% xlabel('U_c(fJ)')
% legend('Data', strcat('fit (\partial\gamma_{nla}/\partialU_c = ', num2str(gamma_nla_sim_vs_Uc_x(1)), "\pm",  num2str(gamma_nla_sim_vs_Uc_bound), "MHz/fJ"))
% 
% figure(300)
% subplot(n,m,9)
% 
% plot(gamma_nla_dash_sim_vs_pd_fit, P_d(slice), gamma_nla_sim)
% 
% figure(301)
% plot(T_analytic_fit, x_dat_coldcav', y_dat_coldcav')
% hold on
% 
% % legend('data', strcat('Fit. (Left lw = ', num2str(kappa_t_left_mhz(1),4),' \pm ', num2str(abs(left_lorentz_lw_bound_mhz),4), ' MHz, \kappa_i = ', num2str(kappa_t_left_mhz(1) - kappa_e_mhz,4), 'MHz.\newlineRight lw = ', num2str(kappa_t_right_mhz,4), ' \pm ', num2str(abs(right_lorentz_lw_bound_mhz),4),' MHz, \kappa_i = ', num2str(kappa_t_right_mhz(1) - kappa_e_mhz,4), 'MHz.\newline\gamma_\beta = ', num2str(kappa_b_mhz,4), 'MHz)'))
% 
% title(titles)
% 
% xlabel("Detuning (MHz)")
% ylabel("Transmission (norm.)")

function [Lorentz_x0, T_analytic_x0] = find_initial_fitting_conditions(x_dat_coldcav, y_dat_coldcav, init_doublet_fitting_params, init_t_analytic_fitting_params, manual_doublet_x0, cutoff)

if manual_doublet_x0 == 0
    dip_indices = find(y_dat_coldcav(:) < cutoff); % Minimum amplitude of the two peaks
    dip_indices_diffs = diff(dip_indices);
    boundary = find(dip_indices_diffs > 5); %>5 to remove noise. Any peak is more than 5 indices wide anyhow
    left_dip = dip_indices(1:boundary); 
    right_dip = dip_indices((boundary+1):end);
    xdat_left_dip = x_dat_coldcav(min(left_dip):max(left_dip));
    xdat_right_dip = x_dat_coldcav(min(right_dip):max(right_dip));
    ydat_left_dip = y_dat_coldcav(min(left_dip):max(left_dip));
    ydat_right_dip = y_dat_coldcav(min(right_dip):max(right_dip));
    cen_left_0 = mean(xdat_left_dip);
    cen_right_0 = mean(xdat_right_dip);
    offset_0 = 1;
    linewidth_left_0 = (max(xdat_left_dip)-min(xdat_left_dip));
    linewidth_right_0 = (max(xdat_right_dip)-min(xdat_right_dip));
    amp_left_0 = (min(ydat_left_dip) - offset_0);
    amp_right_0 = (min(ydat_right_dip) - offset_0);
    Lorentz_x0 = [amp_left_0, linewidth_left_0, cen_left_0, amp_right_0, linewidth_right_0, cen_right_0, offset_0];
    T_analytic_x0 = [100, (cen_right_0+cen_left_0)/2, abs(cen_right_0-cen_left_0)/2, linewidth_left_0, linewidth_right_0];

elseif manual_doublet_x0 == 1
    Lorentz_x0 = init_doublet_fitting_params;
    T_analytic_x0 = init_t_analytic_fitting_params;
end

end
function [Lorentz_x0, T_analytic_x0] = find_initial_fitting_conditions_singlet(x_dat_coldcav, y_dat_coldcav)
    amp_0 = - (max(y_dat_coldcav) - min(y_dat_coldcav));
    linewidth_0 = 500;
    kappa_tot = linewidth_0;
    [value, index] = min(y_dat_coldcav);
    cen_0 = x_dat_coldcav(index);
    offset_0 = 1;
    kappa_e = 100;
    Lorentz_x0 = [amp_0, linewidth_0, cen_0, offset_0];
    T_analytic_x0 = [abs(kappa_e), abs(kappa_tot), cen_0];
end

function [gamma_nla_sim, K_sim, T_min_theory] = simulate_gamma_nla(T_minima, gamma_e, gamma_b, gamma_c, gamma_s)
    
    number_of_runs_to_fit = 16;
    rel_runs = round(linspace(1,number_of_runs_to_fit,number_of_runs_to_fit/4));
    nr_rel = number_of_runs_to_fit;

    line_width = 1;
    colour_max_power = [255,80,10]/255;
    colour_least_power = [41,80,188]/255;
    colour_map = [linspace(colour_least_power(1), colour_max_power(1), nr_rel); linspace(colour_least_power(2), colour_max_power(2), nr_rel);  linspace(colour_least_power(3), colour_max_power(3), nr_rel) ];
  

    gamma_b = abs(gamma_b);
    gamma_nla_sim = zeros(length(T_minima), 1);
    K_sim = zeros(length(T_minima), 1);
    detuning = linspace(-gamma_b/2 - gamma_c - 100,gamma_b/2 + gamma_s + 100,3000);
    for j = 1:length(T_minima)
        T_min_experimental = T_minima(j);
        x = 0;
        gamma_s_tot = gamma_s + gamma_e;
        gamma_c_tot = gamma_c + gamma_e;
        a_c = -sqrt(gamma_e/2).*1./(-gamma_c_tot./2+1i*(detuning + gamma_b/2));
        a_s =  -sqrt(gamma_e/2).*1./(-gamma_s_tot./2+1i*(detuning - gamma_b/2));
        for i = 1: length(detuning)
            T_analytical(i) = get_T(detuning(i),  gamma_e, gamma_b, gamma_c_tot, gamma_s_tot);
        end
        T_min_theory(j) = min(T_analytical);
        error =  abs(T_min_theory(j) - T_min_experimental);
        sign = 1;
        converging_factor = 500;
        counter = 0;
        if j == 1
            disp(strcat("T_min_theory(j) = ",num2str(T_min_theory(j))))
            disp(strcat("T_min_experimental = ", num2str(T_min_experimental)))
        end
        while  abs(error) > 0.0001
            counter = counter + 1;
          
            
            x = x + sign * converging_factor; % x == gamma_nla
            gamma_s_tot = gamma_s + gamma_e + x;
            gamma_c_tot = gamma_c + gamma_e + x;
            for i = 1: length(detuning)
                T_analytical(i) = get_T(detuning(i),  gamma_e, gamma_b, gamma_c_tot, gamma_s_tot);
            end
            T_min_theory(j) = min(T_analytical);

             if (abs(T_min_theory(j) - T_min_experimental)) > error
                sign = -1 * sign;
             end
            error = (abs(T_min_theory(j) - T_min_experimental));
            if converging_factor > 1
                converging_factor = converging_factor/1.2;
            else
                converging_factor = 1;
            end
            if counter > 10000;
                disp("Gamma_nla_sim finding did not converge")
                break
            end


    %         disp(strcat("T_min_experimental = ", num2str(T_min_experimental)))
    %         disp(strcat("error = ", num2str(error)))
    %         disp(strcat("converging factor = ", num2str(converging_factor)))
    %         disp(strcat("x = ", num2str(x)))
    %         disp(strcat("sign = ", num2str(sign)))
        end
        gamma_nla_sim(j) = x;
        if gamma_c > gamma_s
            K_sim(j) = gamma_e/(gamma_s + x);   
        elseif gamma_s > gamma_c 
            K_sim(j) = gamma_e/(gamma_c + x); 
        end
        
        figure(800)
        hold on
        
        if ismember(j,rel_runs)
             plot(detuning./1000, T_analytical,'Color',colour_map(:,j), 'LineWidth', line_width)
        end
        xlabel("Detuning (GHz)")
        ylabel("Transmission (a.u.)")
      
    end
    ax = gca;
    ax.XLim = [min(detuning)/1000 max(detuning)/1000];
    box on
    grid on
end


function T = get_T(wl, gamma_e, gamma_b, gamma_c, gamma_s)
    term_3 = gamma_e/2 * (wl + gamma_b/2)/(gamma_c^2/4 + (wl + gamma_b/2)^2);
    term_4 = gamma_e/2 * (wl - gamma_b/2)/(gamma_s^2/4 + (wl - gamma_b/2)^2);
    term_1 = gamma_e/2 * gamma_c/2/(gamma_c^2/4 + (wl + gamma_b/2)^2);
    term_2 = gamma_e/2 * gamma_s/2/(gamma_s^2/4 + (wl - gamma_b/2)^2);
    T = (-1 + term_1 + term_2)^2 + (term_3 + term_4)^2;
end

function figure_commands()
    grid on;
    ax = gca;

    box on;
end


function[] = fig_po_dur_leg(hf,fname,varargin)
%%% This function requires two inputs, hf(the handle of the figure to be
%%% exported) and fname, the string for the exported pdf file. The figure
%%% must NO?T be docked (use: set(0,'DefaultFigureWindowStyle','normal') to
%%% avoid this). Have fun!
    
set(0,'DefaultFigureWindowStyle','normal')
set(hf, 'WindowStyle','normal')
% STEP 1 - FIND ALL AXES
ch_list_fun = allchild(hf)

for i = 1:numel(ch_list_fun)
    
if strcmp(ch_list_fun(i).Type,'axes') == 1% || strcmp(ch_list_fun(i).Type,'uitable') == 1
    logi(i) = 1;
    ch_list_fun(i).Type
else
    logi(i) = 0;
end
end

log = logical(logi);

ax_list_fun = ch_list_fun(log);
drawnow;
for i = 1:numel(ax_list_fun)
    if strcmp(ax_list_fun(i).Type,'axes') == 1
set(ax_list_fun(i),'Units','centimeters','box','on');
    else
        set(ax_list_fun(i),'Units','centimeters');
    end
end

% STEP 2 - FIND EXTREME POINTS OF AXES AND THEIR POSITIONS
for q = 1:numel(ax_list_fun)
    
    po(q,:) = ax_list_fun(q).Position;
    ti(q,:) = ax_list_fun(q).TightInset;
    
end


right = po(:,1) + po(:,3) + ti(:,3);
top = po(:,2) + po(:,4) + ti(:,4);
bott = po(:,2) - ti(:,2);
left = po(:,1) - ti(:,1);

[r_lim rlimp] = max(right);
[t_lim tlimp] = max(top);
[b_lim blimp] = min(bott);
[l_lim llimp] = min(left);

for q = 1:numel(ax_list_fun)
    set(ax_list_fun(q),'Position',po(q,:) - [po(llimp,1) - ti(llimp,1) po(blimp,2) - ti(blimp,2) 0 0]);
    po2(q,:) = get(ax_list_fun(q),'Position');
end
set(hf, 'Units','centimeters');
drawnow
fpo = get(hf, 'Position');

%%% Find and Accomodate for text objects and legend objects MOSTLY WORKS
htext = [];
htext = findobj(hf, 'Type','text','-or','Type','legend');

if numel(htext) > 0
    HELLO = 5
    for x = 1:numel(htext)
        if strcmp(htext(x).Type,'text') == 1
        set(htext(x),'Units','centimeters')
        txtpo(x,:) = get(htext(x),'Extent');
        par_ax(x) = get(htext(x),'Parent');
        par_po(x,:) = get(par_ax(x),'Position');
        txt_totpo(x,:) = [par_po(x,1)+txtpo(x,1) par_po(x,2)+txtpo(x,2) txtpo(x,3) txtpo(x,4)];
        elseif strcmp(htext(x).Type,'legend') == 1
            set(htext(x),'Units','centimeters')
            txt_totpo(x,:) = htext(x).Position.*[1 1 1.05 1.05]
            
        end
    end
    max_r = max(txt_totpo(:,1) + txt_totpo(:,3));
    min_l = min(txt_totpo(:,1));
    max_t = max(txt_totpo(:,2) + txt_totpo(:,4));
    min_b = min(txt_totpo(:,2));
else
    max_r = 0;
    min_l = 0;
    max_t = 0;
    min_b = 0;
end


% STUPID TIGHT INSET 0 SAFETY FEATURE TO STOP CLIPPING
if ti(rlimp,3) <1E-1
sfty1 = 1;
else
    sfty1 = 0;
end

if ti(tlimp,4) <1E-1
    sfty2 = 1;
else
    sfty2 = 0;
end
sfty1;
sfty2;
gcf = hf;

% STUPID COLORBAR ACCOMADATING STEP
hc = findobj(gcf, 'type','colorbar');
if numel(hc) ~= 0
%     sfty1 = 2.3;
sfty1 = 3

end
% Calculate extra required for text objects
if numel(htext) > 0
    extra_r = ((max_r  - (sfty1+r_lim - (po(llimp,1) - ti(llimp,1)))) > 0)*(max_r  - (sfty1+r_lim - (po(llimp,1) - ti(llimp,1))))+ (min_l < 0)*abs(min_l);
    extra_t = ((max_t  - (sfty2+t_lim - (po(blimp,2) - ti(blimp,2)))) > 0)*(max_t  - (sfty2+t_lim - (po(blimp,2) - ti(blimp,2)))) + (min_b < 0)*abs(min_b);
else
    extra_r = 0;
    extra_t = 0;
end

for q = 1:numel(ax_list_fun)
    set(ax_list_fun(q),'Position',po(q,:) - [po(llimp,1) - ti(llimp,1)-(min_l < 0)*abs(min_l) po(blimp,2) - ti(blimp,2)-(min_b < 0)*abs(min_b) 0 0]);
    po2(q,:) = get(ax_list_fun(q),'Position');
end
set(hf, 'Units','centimeters');
drawnow
fpo = get(hf, 'Position');

% Set close position for figure.
final_fpo = [fpo(1) fpo(2) (extra_r > 0)*extra_r+sfty1+r_lim - (po(llimp,1) - ti(llimp,1)) (extra_t > 0)*extra_t+sfty2+t_lim - (po(blimp,2) - ti(blimp,2))];
final_fpo = [fpo(1) fpo(2) (extra_r > 0)*extra_r+sfty1+r_lim - (po(llimp,1) - ti(llimp,1)) (extra_t > 0)*extra_t+sfty2+t_lim - (po(blimp,2) - ti(blimp,2))];
 set(hf, 'Position',final_fpo);
% 
% 
% % Set Paper size and position
 set(gcf, 'PaperUnits','centimeters');
 set(gcf, 'PaperPositionMode', 'manual');
 set(gcf, 'PaperSize', [final_fpo(3) final_fpo(4)]);
% set(gcf, 'PaperPosition',[(min_l < 0)*abs(min_l) (min_b < 0)*abs(min_b) final_fpo(3)-(min_l < 0)*abs(min_l) final_fpo(4)-(min_b < 0)*abs(min_b)]);
 set(gcf, 'PaperPosition',[0 0 final_fpo(3) final_fpo(4)]);
 figcol = get(gcf, 'Color')
 set(gcf, 'InvertHardCopy','off')%,'Color',[1 1 1])
% Export Figure
set(ax_list_fun,'box','off');

if nargin  - 2 > 0
    if strcmp(varargin{1},'png') == 1
print(strcat(fname),'-dpng','-r600');
    else strcmp(varargin{1},'pdf') == 1
        print(strcat(fname),'-dpdf');
    end
else
print(strcat(fname),'-dpdf');
end
% 
% 
% % savefig(strcat(fname,'_2014'))
% % [status,cmdout] = dos(strcat('cd C:\Program Files\MATLAB\R2014a\bin & matlab -nodesktop -nosplash -r export2014a(''',pwd,''',''',fname,'_2014''',');close all; quit'));
% 
% 
end