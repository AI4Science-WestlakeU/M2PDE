phi_in = './phi.txt'
v = 2.416

[Mesh]
  type = FileMesh
  file = nft.e
[]


[Variables]
  [u]
  []
[]

[AuxVariables]
 [./T]
 [../]
 [./aux_sigma_af]
 [../]
 [flux]
    order = CONSTANT
    family = MONOMIAL
    block = fuel
 [../]
#  [./power]
#  [../]
[]

[Kernels]
  [td]
    type = CoefTimeDerivative
    variable = u
    Coefficient = '${fparse 1 / v}'#
  []
  [diff_fuel]
    type = MatDiffusion
    variable = u
    diffusivity = D_fuel
    block = 'fuel'
  []
  [diff_fluid]
    type = MatDiffusion
    variable = u
    diffusivity = D_fluid
    block = 'fluid'
  []
  [reaction_fuel]
    type = MatReaction
    variable = u
    mob_name = sigma_af_fuel
    block = 'fuel'
  []
  [reaction_fluid]
    type = MatReaction
    variable = u
    mob_name = sigma_af_fluid
    block = 'fluid'
  []
  # [source]
  #   type = BodyForce
  #   variable = u
  #   function = phi_source
  #   block = fuel
  #   value = 1e6
  # []
[]

[AuxKernels]
  [./compute_aux_sigma_af_fuel]
   type = ParsedAux
   variable = aux_sigma_af
   coupled_variables = T
   expression = '2.416 * 583.5 * 1.305 * 1.602 * 0.1 - (13.47 * (T- 560) / (900 - 560) + 7.53) * 2.1479 * 1.602 + 0.185 * 6.6072 * 0.1 - 680.9 * 1.305 * 1.602 * 0.1'
   block = fuel
  [../]
  [./compute_aux_sigma_af_fluid]
   type = ParsedAux
   variable = aux_sigma_af
   coupled_variables = T
   expression = '-(20 + 20 * (T - 560) / (800 - 560))'
   block = fluid
  [../]
  # [./compute_aux_power]
  #  type = ParsedAux
  #  variable = power
  #  coupled_variables = u
  #  expression = '1e8*u'
  # [../]
[]

[Materials]
  [./compute_Dfluid]
    type = GenericConstantMaterial
    prop_names = 'D_fluid'
    prop_values = 0.01
    block = 'fluid'
  [../]
  [./compute_Dfuel]
    type = GenericConstantMaterial
    prop_names = 'D_fuel'
    prop_values = 0.008249
    block = 'fuel'
  [../]
  # [./compute_sigma_af_fuel]
  #   type = GenericFunctionMaterial
  #   prop_names = 'sigma_af_fuel'
  #   prop_values = aux_sigma_af_fuel
  #   block = 'fuel'
  # [../]
  # [./compute_sigma_af_fluid]
  #   type = GenericFunctionMaterial
  #   prop_names = 'sigma_af_fluid'
  #   prop_values = aux_sigma_af_fluid
  #   block = 'fluid'
  # [../]
  [./compute_sigma_af_fuel]
    type = ParsedMaterial
    property_name = sigma_af_fuel
    coupled_variables = T
    expression = '2.416 * 583.5 * 1.305 * 1.602 * 0.1 - (13.47 * (T - 560) / (900 - 560) + 7.53) * 2.1479 * 1.602 + 0.185 * 6.6072 * 0.1 - 680.9 * 1.305 * 1.602 * 0.1'
    block = 'fuel'
  [../]
  [./compute_sigma_af_fluid]
    type = ParsedMaterial
    property_name = sigma_af_fluid
    coupled_variables = T
    expression = '-(20 + 20 * (T - 560) / (800 - 560))'
    block = 'fluid'
  [../]
[]

[Functions]
  [./U_IC]
    type = ParsedFunction
    expression = '2*cos(3.14*(y-0.375)/0.75)'
  [../]
  [./phi_BC]
    type = PiecewiseMultilinear
    data_file = ${phi_in}
  [../]
[]

[BCs]
  [./left]
    type = FunctionDirichletBC
    variable = u
    boundary = 'left'
    function = phi_BC
  [../]

  [./right]
    type = DirichletBC
    variable = u
    boundary = 'right'
    value = 0.0
  [../]

  [./top]
    type = DirichletBC
    variable = u
    value = 0.5
    boundary = top
  [../]

  [./bottom]
    type = DirichletBC
    variable = u
    value = 0.5
    boundary = bottom
  [../]
[]

[ICs]
  [u_ic]
    type = FunctionIC
    variable = 'u'
    function = U_IC
  []
[]

[MultiApps]
  [sub_app]
    type = TransientMultiApp
    positions = '0.0076 0 0'
    input_files = 'fluid.i'
    sub_cycling = true
  []
[]

[Transfers]
  [push_flux]
    type = MultiAppGeneralFieldNearestLocationTransfer

    # Transfer from the sub-app to this app
    to_multi_app = sub_app

    # The name of the variable in the sub-app
    source_variable = flux

    # The name of the auxiliary variable in this app
    variable = flux
    error_on_miss = true
  []

  [pull_temp]
    type = MultiAppGeneralFieldNearestLocationTransfer

    # Transfer from the sub-app to this app
    from_multi_app = sub_app

    # The name of the variable in sub app
    source_variable = T_fluid

    # The name of the auxiliary variable in this
    variable = T
    error_on_miss = true
    # from_blocks = fluid
    to_blocks = fluid
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

