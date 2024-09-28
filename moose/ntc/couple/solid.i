[Mesh]
  type = GeneratedMesh
  dim = 2
  xmin = 0
  xmax = 0.0076
  ymin = 0
  ymax = 0.75
  nx = 8
  ny = 64
  elem_type = QUAD4
[]

[Variables]
  [T]
    initial_condition = 560
  []
[]

[AuxVariables]
  [T_fluid]
    initial_condition = 560
  []
  [flux]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [power]
  [../]
[]

[Kernels]
  [heat_conduction]
    type = HeatConduction
    variable = T
  []
  [./time_diff_m]
    type = HeatConductionTimeDerivative
    variable = T
  [../]
  [source]
    # type = HeatSource
    # variable = T
    # function = power
    type = CoupledForce
    variable = T
    v = power
    coef = 5e7
  []
[]

[AuxKernels]
  [flux_x]
    type = DiffusionFluxAux
    diffusivity = 'thermal_conductivity'
    variable = flux
    diffusion_variable = T
    component = x
    boundary = right
    #execute_on = 'FINAL'
  []
[]

[BCs]
  [./left_symmetry]
    type = NeumannBC
    variable = T
    boundary = 'left'
    value = 0.0
  [../]

  [./right]
    type = FunctorDirichletBC
    variable = T
    boundary = right
    functor = T_fluid
  [../]
[]

[Materials]
  [density_cond]
    type = GenericConstantMaterial
    prop_names = 'density'
    prop_values = '16020'
  []
  [thermal_matrix]
    type = HeatConductionMaterial
    temp = T
    thermal_conductivity_temperature_function = thermal_conductivity
    specific_heat_temperature_function = specific_heat
  []
[]

[Functions]
  [./specific_heat]
    type = ParsedFunction
    expression = '(1.359+0.05812*t+1.086*1e6/t/t)*5'
  [../]
  [./thermal_conductivity]
    type = ParsedFunction
    expression = '17.5*(1-0.223)/(1+0.161)+1.54e-2*(1+0.0061)/(1+0.161)*t+9.38e-6*t*t'
  [../]
  # [./Ty]
  #   type = PiecewiseMultilinear
  #   data_file = ${Ty_file}
  # [../]
  # [./powerxy]
  #   type = PiecewiseMultilinear
  #   data_file = ${power_file}
  # [../]
[]

[MultiApps]
  [sub_app]
    type = TransientMultiApp
    positions = '0.0 0 0'
    input_files = 'neutron.i'
    sub_cycling = true
  []
[]

[Transfers]
  [pull_T]
    type = MultiAppGeneralFieldNearestLocationTransfer

    # Transfer from the sub-app to this app
    from_multi_app = sub_app

    # The name of the variable in the sub-app
    source_variable = T

    # The name of the auxiliary variable in this app
    variable = T_fluid
    from_blocks = fluid
    # to_blocks = fluid
  []

  [pull_power]
    type = MultiAppGeneralFieldNearestLocationTransfer
    from_multi_app = sub_app
    source_variable = u
    variable = power
    error_on_miss = true
  []

  [push_flux]
    type = MultiAppGeneralFieldNearestLocationTransfer

    # Transfer to the sub-app from this app
    to_multi_app = sub_app

    # The name of the variable in this app
    source_variable = flux

    # The name of the auxiliary variable in the sub-app
    variable = flux
    error_on_miss = true
  []

  [push_T]
    type = MultiAppGeneralFieldNearestLocationTransfer
    to_multi_app = sub_app
    source_variable = T
    variable = T
    error_on_miss = true
    # from_blocks = fuel
    to_blocks = fuel
  []
[]

[Executioner]
  type = Transient
  start_time = 0
  end_time = 5
  #num_steps = 32
  #dt = 0.1
[]

[Outputs]
  [exodus]
    type = Exodus
    sync_only = true
    sync_times = '0. 0.3125 0.625 0.9375 1.25 1.5625 1.875 2.1875 2.5 2.8125 3.125 3.4375 3.75 4.0625 4.375 4.6875 5.'
  []
[]

