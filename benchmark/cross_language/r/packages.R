required <- c("bench", "copula", "jsonlite", "RcppTOML", "renv")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
    install.packages(missing, repos = "https://cloud.r-project.org")
}
