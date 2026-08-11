args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
project <- normalizePath(dirname(sub("^--file=", "", file_arg[[1]])))
renv::snapshot(project = project, type = "explicit", prompt = FALSE)
