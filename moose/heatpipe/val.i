[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Mesh]
  type = FileMesh
  file = val_mesh.e  #
[]

[Variables]
  [T]
    initial_condition = 948
  []
[]
[AuxVariables]
  [flux_BC]
  []
[]

[Kernels]
  [heat_conduction]
    type = HeatConduction
    variable = T
    use_displaced_mesh = true
  []
[]

[Physics/SolidMechanics/QuasiStatic]
  [all]
    add_variables = true
    strain = FINITE
    automatic_eigenstrain_names = true
    planar_formulation = PLANE_STRAIN
    generate_output = 'vonmises_stress strain_xx strain_yy strain_zz stress_xx stress_yy stress_zz'
  []
[]

[Materials]
  [./youngs_modulus_matrix]
    type = ParsedMaterial
    property_name = youngs_modulus_matrix
    coupled_variables = 'T'
    expression = '(211600-51.73*(T-273)-0.01928*(T-273)*(T-273))*1E6'
    outputs = exodus
  [../]
  [./elasticity_tensor_matrix]
    type = ComputeVariableIsotropicElasticityTensor
    args = T
    youngs_modulus = youngs_modulus_matrix
    poissons_ratio = 0.21
  [../]

  [thermal_matrix]
    type = HeatConductionMaterial
    temp = T
    thermal_conductivity_temperature_function = thermal_conductivity_function_matrix
    use_displaced_mesh = true
    # block = '101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116'
  []

  [./thermal_expansion_matrix]
    type = ComputeMeanThermalExpansionFunctionEigenstrain
    thermal_expansion_function = cte_func_mean_matrix
    thermal_expansion_function_reference_temperature = 293
    stress_free_temperature = 0.0
    temperature = T
    eigenstrain_name = eigenstrain
    # block = '101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116'
  [../]
  [stress]
    type = ComputeFiniteStrainElasticStress
  []
[]

[Functions]
  [./cte_func_mean_matrix]
    type = ParsedFunction
    expression = 'if(t>1273,5e-6,1e-6*(-1.8276+0.0178*t-1.5544e-5*t*t+4.5246e-9*t*t*t))'
  [../]
  [./thermal_conductivity_function_matrix]
    type = ParsedFunction
    expression = '9.2+0.0175*(t-273)-2e-6*(t-273)*(t-273)'
  [../]


[]

[BCs]
  [heatpipe]
    type = DirichletBC
    variable = T
    value = 948
    boundary = heatpipe
  []
  [./left_symmetry]
    type = NeumannBC
    variable = T
    boundary = 'left'
    value = 0.0
  [../]
  [./bottom_symmetry]
    type = NeumannBC
    variable = T
    boundary = 'bottom'
    value = 0.0
  [../]
  [./flux]
    type = FunctorNeumannBC#NeumannBC
    variable = T
    boundary = 'matrixin_b'
    functor = flux_BC #
  [../]
  [./Pressure]
    [./inter]
      boundary = 'matrixin_b'
      factor = 1
      #variable = disp_x
      displacements = "disp_x disp_y"
      function = 2e6
    [../]
  [../]

  [./InclinedNoDisplacementBC]

    [./right]
      boundary = right
      penalty = 1e15
      displacements = "disp_x disp_y"
    [../]

    [./left]
      boundary = left
      penalty = 1e15
      displacements = "disp_x disp_y"
    [../]

  [../]




[]

[ICs]

  [./ic_flux_101]
    type = ConstantIC
    variable = flux_BC
    value = 294668.9697687741
    block = 101
  [../]

  [./ic_flux_102]
    type = ConstantIC
    variable = flux_BC
    value = 404207.48063133424
    block = 102
  [../]

  [./ic_flux_103]
    type = ConstantIC
    variable = flux_BC
    value = 335517.75721379626
    block = 103
  [../]

  [./ic_flux_104]
    type = ConstantIC
    variable = flux_BC
    value = 683805.3224320388
    block = 104
  [../]

  [./ic_flux_105]
    type = ConstantIC
    variable = flux_BC
    value = 727621.1528935636
    block = 105
  [../]

  [./ic_flux_106]
    type = ConstantIC
    variable = flux_BC
    value = 424200.779625628
    block = 106
  [../]

  [./ic_flux_107]
    type = ConstantIC
    variable = flux_BC
    value = 555727.8509678291
    block = 107
  [../]

  [./ic_flux_108]
    type = ConstantIC
    variable = flux_BC
    value = 703366.3798771359
    block = 108
  [../]

  [./ic_flux_109]
    type = ConstantIC
    variable = flux_BC
    value = 356036.8396807039
    block = 109
  [../]

  [./ic_flux_110]
    type = ConstantIC
    variable = flux_BC
    value = 408982.5040916194
    block = 110
  [../]

  [./ic_flux_111]
    type = ConstantIC
    variable = flux_BC
    value = 583256.8852234726
    block = 111
  [../]

  [./ic_flux_112]
    type = ConstantIC
    variable = flux_BC
    value = 761202.8469475714
    block = 112
  [../]

  [./ic_flux_113]
    type = ConstantIC
    variable = flux_BC
    value = 733433.2482415673
    block = 113
  [../]

  [./ic_flux_114]
    type = ConstantIC
    variable = flux_BC
    value = 720161.0755083198
    block = 114
  [../]

  [./ic_flux_115]
    type = ConstantIC
    variable = flux_BC
    value = 430976.6889581246
    block = 115
  [../]

  [./ic_flux_116]
    type = ConstantIC
    variable = flux_BC
    value = 596351.2871466015
    block = 116
  [../]

  [./ic_flux_117]
    type = ConstantIC
    variable = flux_BC
    value = 300802.0349157377
    block = 117
  [../]

  [./ic_flux_118]
    type = ConstantIC
    variable = flux_BC
    value = 414328.50636636384
    block = 118
  [../]

  [./ic_flux_119]
    type = ConstantIC
    variable = flux_BC
    value = 570029.9713513975
    block = 119
  [../]

  [./ic_flux_120]
    type = ConstantIC
    variable = flux_BC
    value = 339994.5223299095
    block = 120
  [../]

  [./ic_flux_121]
    type = ConstantIC
    variable = flux_BC
    value = 550603.7397915875
    block = 121
  [../]

  [./ic_flux_122]
    type = ConstantIC
    variable = flux_BC
    value = 723424.8321061527
    block = 122
  [../]

  [./ic_flux_123]
    type = ConstantIC
    variable = flux_BC
    value = 524145.4816860942
    block = 123
  [../]

  [./ic_flux_124]
    type = ConstantIC
    variable = flux_BC
    value = 338391.24151988537
    block = 124
  [../]

  [./ic_flux_125]
    type = ConstantIC
    variable = flux_BC
    value = 750998.8907615264
    block = 125
  [../]

  [./ic_flux_126]
    type = ConstantIC
    variable = flux_BC
    value = 477173.3323906947
    block = 126
  [../]

  [./ic_flux_127]
    type = ConstantIC
    variable = flux_BC
    value = 504514.9958648608
    block = 127
  [../]

  [./ic_flux_128]
    type = ConstantIC
    variable = flux_BC
    value = 592862.296095565
    block = 128
  [../]

  [./ic_flux_129]
    type = ConstantIC
    variable = flux_BC
    value = 656243.6657427398
    block = 129
  [../]

  [./ic_flux_130]
    type = ConstantIC
    variable = flux_BC
    value = 258664.4693982481
    block = 130
  [../]

  [./ic_flux_131]
    type = ConstantIC
    variable = flux_BC
    value = 217543.94512146045
    block = 131
  [../]

  [./ic_flux_132]
    type = ConstantIC
    variable = flux_BC
    value = 720568.2538484393
    block = 132
  [../]

  [./ic_flux_133]
    type = ConstantIC
    variable = flux_BC
    value = 571014.1108643459
    block = 133
  [../]

  [./ic_flux_134]
    type = ConstantIC
    variable = flux_BC
    value = 772903.2004458384
    block = 134
  [../]

  [./ic_flux_135]
    type = ConstantIC
    variable = flux_BC
    value = 316563.02515251003
    block = 135
  [../]

  [./ic_flux_136]
    type = ConstantIC
    variable = flux_BC
    value = 320381.55441379925
    block = 136
  [../]

  [./ic_flux_137]
    type = ConstantIC
    variable = flux_BC
    value = 649551.2640833164
    block = 137
  [../]

  [./ic_flux_138]
    type = ConstantIC
    variable = flux_BC
    value = 213379.0761041243
    block = 138
  [../]

  [./ic_flux_139]
    type = ConstantIC
    variable = flux_BC
    value = 450816.2095540936
    block = 139
  [../]

  [./ic_flux_140]
    type = ConstantIC
    variable = flux_BC
    value = 667057.4183549825
    block = 140
  [../]

  [./ic_flux_141]
    type = ConstantIC
    variable = flux_BC
    value = 375223.4357565421
    block = 141
  [../]

  [./ic_flux_142]
    type = ConstantIC
    variable = flux_BC
    value = 718337.6264003261
    block = 142
  [../]

  [./ic_flux_143]
    type = ConstantIC
    variable = flux_BC
    value = 598835.5561871505
    block = 143
  [../]

  [./ic_flux_144]
    type = ConstantIC
    variable = flux_BC
    value = 285642.01829264534
    block = 144
  [../]

  [./ic_flux_145]
    type = ConstantIC
    variable = flux_BC
    value = 423779.94161605765
    block = 145
  [../]

  [./ic_flux_146]
    type = ConstantIC
    variable = flux_BC
    value = 681194.0602452112
    block = 146
  [../]

  [./ic_flux_147]
    type = ConstantIC
    variable = flux_BC
    value = 261802.73277457897
    block = 147
  [../]

  [./ic_flux_148]
    type = ConstantIC
    variable = flux_BC
    value = 552325.9135566512
    block = 148
  [../]

  [./ic_flux_149]
    type = ConstantIC
    variable = flux_BC
    value = 474750.4023580325
    block = 149
  [../]

  [./ic_flux_150]
    type = ConstantIC
    variable = flux_BC
    value = 335845.2317051966
    block = 150
  [../]

  [./ic_flux_151]
    type = ConstantIC
    variable = flux_BC
    value = 324577.7497897311
    block = 151
  [../]

  [./ic_flux_152]
    type = ConstantIC
    variable = flux_BC
    value = 283471.6900043059
    block = 152
  [../]

  [./ic_flux_153]
    type = ConstantIC
    variable = flux_BC
    value = 485941.3992516879
    block = 153
  [../]

  [./ic_flux_154]
    type = ConstantIC
    variable = flux_BC
    value = 200137.4778023388
    block = 154
  [../]

  [./ic_flux_155]
    type = ConstantIC
    variable = flux_BC
    value = 517442.0475929167
    block = 155
  [../]

  [./ic_flux_156]
    type = ConstantIC
    variable = flux_BC
    value = 657399.2188427348
    block = 156
  [../]

  [./ic_flux_157]
    type = ConstantIC
    variable = flux_BC
    value = 597367.9976120575
    block = 157
  [../]

  [./ic_flux_158]
    type = ConstantIC
    variable = flux_BC
    value = 297352.88818336645
    block = 158
  [../]

  [./ic_flux_159]
    type = ConstantIC
    variable = flux_BC
    value = 679717.989800913
    block = 159
  [../]

  [./ic_flux_160]
    type = ConstantIC
    variable = flux_BC
    value = 568273.2466395549
    block = 160
  [../]

  [./ic_flux_161]
    type = ConstantIC
    variable = flux_BC
    value = 375999.33896832983
    block = 161
  [../]

  [./ic_flux_162]
    type = ConstantIC
    variable = flux_BC
    value = 540444.8943351862
    block = 162
  [../]

  [./ic_flux_163]
    type = ConstantIC
    variable = flux_BC
    value = 603147.5651880725
    block = 163
  [../]

  [./ic_flux_164]
    type = ConstantIC
    variable = flux_BC
    value = 368486.3671323827
    block = 164
  [../]

  [./ic_disp_x]
      type = FunctionIC
      variable = 'disp_x'
      function = 0
  [../]
  [./ic_disp_y]
      type = FunctionIC
      variable = 'disp_y'
      function = 0
  [../]
[]

[Preconditioning]
  [smp]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Steady
  petsc_options_iname = '-pc_type'
  petsc_options_value = 'lu'
  # l_max_its = 20
  # nl_max_its = 5
  #end_time = 5
  #dt = 1
[]

[Outputs]
  [exodus]
    type = Exodus
    elemental_as_nodal = true
  []
[]
