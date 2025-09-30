using TestItemRunner

# Remember that you can filter tests you want to be ran by ]test here : simply filter them like follows: 
# @run_package_tests filter=ti->(:GumbelBarnettCopula in ti.tags || :ArchimedeanCopula in ti.tags || :FrankCopula in ti.tags)
# you can add verbose=true here 

@run_package_tests filter=ti->(
    :FGMCopula in ti.tags ||
    :PlackettCopula in it.tags ||
    :SurvivalCopula in ti.tags ||
    :GaussianCopula in ti.tags ||
    :TCopula in ti.tags ||
    :GalambosCopula in ti.tags ||
    :AsymMixedCopula in ti.tags ||
    :CuadrasAugeCopula in ti.tags ||
    :HuslerReisCopula in ti.tags ||
    :MixedCopula in ti.tags ||
    :EmpiricalEVCopula in ti.tags ||
    :ArchimaxCopula in ti.tags
)