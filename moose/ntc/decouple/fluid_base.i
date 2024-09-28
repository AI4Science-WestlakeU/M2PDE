flux_file = %(flux_file)s
# Fluid properties
rho = 11096
# Operating conditions
v_inlet = 1
T_inlet = 560
p_outlet = 0

# Numerical scheme
advected_interp_method = 'average'
velocity_interp_method = 'rc'

[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 2
    xmin = 0
    xmax = 0.0114
    ymin = 0
    ymax = 0.75
    nx = 12
    ny = 64
  []
[]

[GlobalParams]
  rhie_chow_user_object = 'rc'
[]

[UserObjects]
  [rc]
    type = INSFVRhieChowInterpolator
    u = vel_x
    v = vel_y
    pressure = pressure
  []
[]

[Variables]
  [vel_x]
    type = INSFVVelocityVariable
    initial_condition = ${v_inlet}
  []
  [vel_y]
    type = INSFVVelocityVariable
    initial_condition = 1e-12
  []
  [pressure]
    type = INSFVPressureVariable
  []
  [T_fluid]
    type = INSFVEnergyVariable
    initial_condition = ${T_inlet}
  []
[]

[AuxVariables]
  [./aux_mu]
    family = MONOMIAL
    order = FIRST
  [../]
  [./aux_k]
    family = MONOMIAL
    order = FIRST
  [../]
  [./aux_cp]
    family = MONOMIAL
    order = FIRST
  [../]
  [flux]
    order = CONSTANT
    family = MONOMIAL
    initial_condition = 0
  []
[]

[FVKernels]
  [mass]
    type = INSFVMassAdvection
    variable = pressure
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
  []
  [u_time]
    type = INSFVMomentumTimeDerivative
    variable = vel_x
    rho = ${rho}
    momentum_component = 'x'
  []
  [u_advection]
    type = INSFVMomentumAdvection
    variable = vel_x
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
    momentum_component = 'x'
  []
  [u_viscosity]
    type = INSFVMomentumDiffusion
    variable = vel_x
    mu = aux_mu
    momentum_component = 'x'
  []
  [u_pressure]
    type = INSFVMomentumPressure
    variable = vel_x
    momentum_component = 'x'
    pressure = pressure
  []

  [v_time]
    type = INSFVMomentumTimeDerivative
    variable = vel_y
    rho = ${rho}
    momentum_component = 'y'
  []
  [v_advection]
    type = INSFVMomentumAdvection
    variable = vel_y
    advected_interp_method = ${advected_interp_method}
    velocity_interp_method = ${velocity_interp_method}
    rho = ${rho}
    momentum_component = 'y'
  []
  [v_viscosity]
    type = INSFVMomentumDiffusion
    variable = vel_y
    mu = aux_mu
    momentum_component = 'y'
  []
  [v_pressure]
    type = INSFVMomentumPressure
    variable = vel_y
    momentum_component = 'y'
    pressure = pressure
  []

  [energy_time]
    type = INSFVEnergyTimeDerivative
    variable = T_fluid
    rho = ${rho}
    dh_dt = dh_dt
  []
  [energy_advection]
    type = INSFVEnergyAdvection
    variable = T_fluid
    velocity_interp_method = ${velocity_interp_method}
    advected_interp_method = ${advected_interp_method}
  []
  [energy_diffusion]
    type = FVDiffusion
    variable = T_fluid
    coeff = aux_k
  []
[]

[AuxKernels]
  [./compute_aux_mu]
    type = ParsedAux
    variable = aux_mu
    coupled_variables = 'T_fluid'
    function = '4.94e-4*exp(754.1/T_fluid)'
  [../]
  [./compute_aux_k]
    type = ParsedAux
    variable = aux_k
    coupled_variables = 'T_fluid'
    function = '3.61+1.517e-2*T_fluid-1.741e-6*T_fluid*T_fluid'
  [../]
  [./compute_aux_cp]
    type = ParsedAux
    variable = aux_cp
    coupled_variables = 'T_fluid'
    function = '159-2.72e-2*T_fluid+7.12e-6*T_fluid*T_fluid'
  [../]
  [flux]
    type = FunctorAux
    functor = f_flux
    variable = flux
  []
[]

[FVBCs]
  [inlet-v]
    type = INSFVInletVelocityBC
    boundary = 'bottom'
    variable = vel_y
    function = inlet_v
  []
  [inlet-u]
    type = INSFVInletVelocityBC
    boundary = 'bottom'
    variable = vel_x
    function = 0
  []

  [inlet-T]
    type = FVDirichletBC
    variable = T_fluid
    value = '${fparse T_inlet}'#
    boundary = 'bottom'
  []

  [T_left]
    type = FVFunctorNeumannBC
    variable = T_fluid
    boundary = 'left'
    functor = flux
    factor = 1
  []

  [no-slip-u]
    type = INSFVNoSlipWallBC
    boundary = 'left'
    variable = vel_x
    function = 0
  []
  [no-slip-v]
    type = INSFVNoSlipWallBC
    boundary = 'left'
    variable = vel_y
    function = 0
  []

  [symmetry-u]
    type = INSFVSymmetryVelocityBC
    boundary = 'right'
    variable = vel_x
    u = vel_x
    v = vel_y
    mu = aux_mu
    momentum_component = 'x'
  []
  [symmetry-v]
    type = INSFVSymmetryVelocityBC
    boundary = 'right'
    variable = vel_y
    u = vel_x
    v = vel_y
    mu = aux_mu
    momentum_component = 'y'
  []
  [symmetry-p]
    type = INSFVSymmetryPressureBC
    boundary = 'right'
    variable = pressure
  []

  [outlet_u]
    type = INSFVMomentumAdvectionOutflowBC
    variable = vel_x
    u = vel_x
    v = vel_y
    boundary = 'top'
    momentum_component = 'x'
    rho = ${rho}
  []
  [outlet_v]
    type = INSFVMomentumAdvectionOutflowBC
    variable = vel_y
    u = vel_x
    v = vel_y
    boundary = 'top'
    momentum_component = 'y'
    rho = ${rho}
  []
  [outlet_p]
    type = INSFVOutletPressureBC
    boundary = 'top'
    variable = pressure
    function = '${p_outlet}'
  []
[]

[FunctorMaterials]
  [functor_constants]
    type = ADGenericFunctorMaterial
    prop_names = 'cp k mu'
    prop_values = 'aux_cp aux_k aux_mu'
  []
  [ins_fv]
    type = INSFVEnthalpyFunctorMaterial
    rho = ${rho}
    temperature = 'T_fluid'
  []
[]


[Functions]
  [./inlet_v]
    type = ParsedFunction
    expression = '0.4'
  [../]
  [./f_flux]
    type = PiecewiseMultilinear
    data_file = ${flux_file}
  [../]
[]

[Executioner]
  type = Transient
  solve_type = 'NEWTON'
  petsc_options_iname = '-pc_type -pc_factor_shift_type'
  petsc_options_value = 'lu NONZERO'
  line_search = 'none'
  nl_rel_tol = 1e-8
  nl_abs_tol = 1e-8
  nl_max_its = 20

  l_tol = 1e-5
  l_max_its = 100
  start_time = 0
  end_time = 5
  # num_steps = 32
[]

[Outputs]
  [exodus]
    type = Exodus
    sync_only = true
    sync_times = '0. 0.3125 0.625 0.9375 1.25 1.5625 1.875 2.1875 2.5 2.8125 3.125 3.4375 3.75 4.0625 4.375 4.6875 5.'
  []
[]
