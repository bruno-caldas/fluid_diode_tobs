function opt = tobs_from_matlab(nvar, x_L, x_U, cst_num, acst_L, acst_U, obj_fun, obj_dfun, cst_fval, jacobian, iteration, epsilons, rho, volume_constraint, flip_limits)

addpath(genpath('BrunoDoc/FEA'))
addpath(genpath('BrunoDoc/Meshes'))
addpath(genpath('BrunoDoc/TopOpt'))
disp(' ')
disp('         *****************************')
disp('         **   Topology Pi - START   **')
disp('         *****************************')
disp(' ')

%% --------------------------------------------------------------------- %%
%                              ** Input **                                %
%-------------------------------------------------------------------------%

% Optimization parameters
%radius = 6;                  % Filter radius in length unit
%rho_min = 0.001^3;           % Minimum density (for void elements)

% volume_constraint = 0; % Compliance (Nm) constraint
% flip_limits = 0.01;               % Flip limits
% flip_limits = 0.5;               % Flip limits
% flip_limits = 0.2;               % Flip limits

%% --------------------------------------------------------------------- %%
%                         ** Problem set up **                            %
%-------------------------------------------------------------------------%
% Prepare TOBS
tobs = TOBS(volume_constraint, epsilons, flip_limits, nvar);


%% --------------------------------------------------------------------- %%
%                           ** Optimization **                            %
%-------------------------------------------------------------------------%

disp(' ')
disp('         *** Optimization loop *** ')
disp(' ')

tobs.design_variables = rho
tobs.objective = obj_fun;
tobs.objective_sensitivities = obj_dfun;

% Print optimization status on the screen
disp([' It.: ' sprintf('%3i',iteration) '  Obj.: ' sprintf('%5.4f',full(tobs.objective))...
    '  Vol.: ' ])

% Optimization loop %

% Convergence identifiers
is_converged = 0;
difference = 1;

loop = iteration;

% Constraint (compliance) and sensitivities
% tobs.constraints = speye (1)*cst_fval;
tobs.constraints = cst_fval;
tobs.constraints_sensitivities = jacobian;
disp('constraints')
disp(cst_fval)
disp('')
disp('volume_constraint')
disp(volume_constraint)
disp('')
disp('epsilons')
disp(epsilons)
disp('')
disp('jacobian')
disp(size(jacobian))
disp('')

[tobs, PythonObjCoeff, ...
		    PythonConstCoeff, PythonRelaxedLimits, ...
		    PythonLowerLimits, PythonUpperLimits, PythonnDesignVariables] = SolveWithILP(tobs);

% Storing optimization history
tobs.history(loop+1,1) = tobs.objective;
% tobs.history(loop+1,2) = tobs.constraints; %FIXME: NAO FUNCIONA COM VARIAS RESTRICOES DE VOLUME

% Finite Element analysis
opt = {tobs.design_variables, PythonObjCoeff, ...
		    PythonConstCoeff, PythonRelaxedLimits, ...
		    PythonLowerLimits, PythonUpperLimits, PythonnDesignVariables};

end
