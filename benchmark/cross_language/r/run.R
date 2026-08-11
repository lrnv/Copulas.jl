args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", file_arg[[1]]))
root <- normalizePath(file.path(dirname(script_path), ".."))

# setup-renv restores into an isolated project library. Explicit loading keeps
# this runner independent of its working directory and of R profile processing.
if (nzchar(Sys.getenv("RENV_PATHS_ROOT"))) {
    if (!requireNamespace("renv", quietly = TRUE)) stop("renv is required in CI")
    renv::load(project = file.path(root, "r"), quiet = TRUE)
}

suppressPackageStartupMessages({
    library(bench)
    library(copula)
    library(jsonlite)
    library(RcppTOML)
})

spec <- RcppTOML::parseTOML(file.path(root, "cases.toml"))

mode <- Sys.getenv("CROSS_BENCH_MODE", "full")
if (!mode %in% c("smoke", "full")) stop("CROSS_BENCH_MODE must be smoke or full")
output <- Sys.getenv("CROSS_BENCH_OUTPUT", file.path(root, "results", "r.json"))
seed <- as.integer(spec$suite$seed)
validation_points <- as.integer(spec$suite$validation_points)
repetitions <- as.integer(spec$suite[[if (mode == "smoke") "smoke_repetitions" else "full_repetitions"]])

make_model <- function(case, fitting = FALSE) {
    family <- case$family
    d <- as.integer(case$dimension)
    parameter <- as.numeric(case$parameter)
    if (family == "clayton") return(claytonCopula(parameter, dim = d))
    if (family == "gumbel") return(gumbelCopula(parameter, dim = d))
    if (family == "gaussian") {
        if (fitting) return(normalCopula(dim = d, dispstr = "un"))
        return(normalCopula(parameter, dim = d, dispstr = "ex"))
    }
    stop(sprintf("Unsupported family: %s", family))
}

read_input <- function(case) {
    path <- file.path(root, case$input)
    rows <- as.matrix(read.csv(path, header = FALSE, check.names = FALSE))
    n_key <- if (mode == "smoke") "smoke_n" else "n"
    n <- as.integer(case[[n_key]])
    d <- as.integer(case$dimension)
    if (nrow(rows) < n || ncol(rows) < d) stop(sprintf("Invalid fixture dimensions: %s", path))
    rows[seq_len(n), seq_len(d), drop = FALSE]
}

prepare <- function(case) {
    operation <- case$operation
    n_key <- if (mode == "smoke") "smoke_n" else "n"
    n <- as.integer(case[[n_key]])

    if (operation == "sample") {
        model <- make_model(case)
        f <- function() rCopula(n, model)
        set.seed(seed + 1L)
        check <- rCopula(min(n, 1000L), model)
        valid <- all(is.finite(check)) && all(check >= 0) && all(check <= 1)
        return(list(f = f, kind = "stochastic_summary", values = colMeans(check), valid = valid))
    }
    if (operation == "logdensity") {
        model <- make_model(case)
        points <- read_input(case)
        f <- function() dCopula(points, model, log = TRUE)
        validation_rows <- seq_len(min(validation_points, nrow(points)))
        values <- dCopula(points[validation_rows, , drop = FALSE], model, log = TRUE)
        return(list(f = f, kind = "numeric", values = values, valid = all(is.finite(values))))
    }
    if (operation == "pseudos") {
        data <- read_input(case)
        f <- function() pobs(data)
        transformed <- pobs(data)
        rows <- seq_len(min(validation_points, nrow(transformed)))
        values <- as.numeric(t(transformed[rows, , drop = FALSE]))
        valid <- all(is.finite(transformed)) && all(transformed > 0) && all(transformed < 1)
        return(list(f = f, kind = "numeric", values = values, valid = valid))
    }
    if (operation == "fit_itau") {
        data <- read_input(case)
        f <- function() fitCopula(gumbelCopula(dim = 2), data, method = "itau", estimate.variance = FALSE)
        values <- as.numeric(coef(f()))
        return(list(f = f, kind = "numeric", values = values, valid = all(is.finite(values))))
    }
    if (operation == "fit_mle") {
        data <- read_input(case)
        d <- as.integer(case$dimension)
        start <- rep(0.1, d * (d - 1) / 2)
        f <- function() fitCopula(
            make_model(case, fitting = TRUE), data,
            method = "ml", start = start, estimate.variance = FALSE
        )
        fitted <- f()
        sigma <- getSigma(fitted@copula)
        values <- unlist(lapply(seq.int(2, d), function(j) sigma[seq_len(j - 1), j]))
        return(list(f = f, kind = "numeric", values = values, valid = all(is.finite(values))))
    }
    stop(sprintf("Unsupported operation: %s", operation))
}

measure <- function(f) {
    invisible(f())
    invisible(gc())
    result <- bench::mark(
        f(),
        iterations = repetitions,
        check = FALSE,
        filter_gc = FALSE,
        memory = TRUE
    )
    list(
        median_ns = as.numeric(result$median[[1]], units = "secs") * 1e9,
        minimum_ns = as.numeric(result$min[[1]], units = "secs") * 1e9,
        memory_bytes = as.numeric(result$mem_alloc[[1]]),
        allocations = NA,
        iterations = as.integer(result$n_itr[[1]])
    )
}

set.seed(seed)
results <- lapply(spec$cases, function(case) {
    message(sprintf("Benchmarking %s", case$name))
    prepared <- prepare(case)
    if (!isTRUE(prepared$valid)) stop(sprintf("Local validation failed for %s", case$name))
    list(
        name = case$name,
        operation = case$operation,
        timing = measure(prepared$f),
        validation = list(
            kind = prepared$kind,
            valid = prepared$valid,
            values = unname(as.numeric(prepared$values))
        )
    )
})

record <- list(
    schema_version = as.integer(spec$suite$schema_version),
    language = "R",
    mode = mode,
    generated_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
    runtime_version = R.version.string,
    package_versions = list(
        copula = as.character(packageVersion("copula")),
        bench = as.character(packageVersion("bench"))
    ),
    environment = list(
        os = Sys.info()[["sysname"]],
        architecture = R.version$arch,
        threads = as.integer(Sys.getenv("OMP_NUM_THREADS", "1")),
        git_commit = Sys.getenv("GITHUB_SHA", "local")
    ),
    benchmarks = results
)

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(record, output, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null")
message(sprintf("Wrote %s", output))
