using TestItemRunner

# Remember that you can filter tests you want to be ran by ]test here : simply filter them like follows: 
# @run_package_tests filter=ti->(:GumbelBarnettCopula in ti.tags || :ArchimedeanCopula in ti.tags || :FrankCopula in ti.tags)
# you can add verbose=true here 

@run_package_tests filter=ti->(:Fitting in ti.tags)