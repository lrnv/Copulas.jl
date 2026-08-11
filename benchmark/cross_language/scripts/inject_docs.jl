length(ARGS) == 2 || error("usage: inject_docs.jl COMPARISON_MARKDOWN DOCS_MARKDOWN")

const RESULTS_PATH, DOCS_PATH = ARGS
const BEGIN_MARKER = "<!-- BEGIN JULIA_VS_R_RESULTS -->"
const END_MARKER = "<!-- END JULIA_VS_R_RESULTS -->"

results = read(RESULTS_PATH, String)
docs = read(DOCS_PATH, String)

startswith(results, "# Julia and R benchmark comparison") ||
    error("Unexpected comparison report heading in $RESULTS_PATH")
length(findall(BEGIN_MARKER, docs)) == 1 ||
    error("Expected exactly one results block in $DOCS_PATH")

# The report is also a standalone Actions summary. Inside the manual page,
# demote its heading to the appropriate level.
embedded = replace(results, r"^# Julia and R benchmark comparison\r?\n" =>
                            "### Results from this documentation build\n"; count=1)
replacement = string(BEGIN_MARKER, "\n\n", strip(embedded), "\n\n", END_MARKER)
pattern = Regex(string("(?s)", BEGIN_MARKER, ".*?", END_MARKER))
updated = replace(docs, pattern => replacement; count=1)
updated == docs && error("No results block was replaced in $DOCS_PATH")

write(DOCS_PATH, updated)
println("Embedded $RESULTS_PATH in $DOCS_PATH")
