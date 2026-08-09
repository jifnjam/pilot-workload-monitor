% compute_stats.m
% 
%
% Purpose:
% 1. Check Python's welch-based theta band power against 
%   MATLAB's bandpower() function on the same signal.
%   This will validate that the the Python extraction pipeline
%   is working correctly.
% 2. Run a one-way ANOVA to test whether the theta power differs
%    across different workload levels using the Python-built feature table.

cd(fileparts(mfilename(fullfile('fullpath'))));
dir = fullfile('..', '..', '..', 'data', 'processed');

%% Checking one single epoch

crosscheck = load(fullfile(dir, 'matlab_check.mat'));

matlab_theta_power = bandpower(crosscheck.example_signal, crosscheck.sfreq, [4 8]);
python_theta_power = crosscheck.python_theta_power;
pct_diff = (abs(matlab_theta_power - python_theta_power) / python_theta_power) * 100;

fprintf('Cross-checking single epoch theta power\n');
fprintf('Python (Welch-based): %.4e\n', python_theta_power);
fprintf('MATLAB: %.4e\n', matlab_theta_power);
fprintf('Relative difference: %.2f%%\n', pct_diff);
fprintf('Reminder: do not expect equality in results\n')

%% One-Way ANOVA across the workload levels

features = readtable(fullfile(dir, 'theta_power_features.csv'));

[p, tbl, stats] = anova1(features.theta_power, features.test, 'off');

fprintf('One-Way ANOVA across the workload levels\n');
fprintf('p-value is: %.4f\n', p);
if p < 0.05
     fprintf('The p-value is less than 0.05\nThis pattern did not occur by chance and therefore is statistically important.');
else
     fprintf('The p-value is greater than or equal to 0.05 so there is no strong evidence of theta power differing across the conditions in the data.');

end

%% Saving results for exporting and manipulation with Python


results = table(p, pct_diff, 'VariableNames', {'anova_p_value', 'crosscheck_pct_diff'});
writetable(results, fullfile(dir, 'matlab_anova_results.csv'));
fprintf('\nSaved file to matlab_anova_results.csv\n')