using Markdown

const BENCHMARK_PAGE = joinpath(@__DIR__, "..", "docs", "src", "dev", "benchmarks.md")
const BLOCK_PATTERN = r"(?s)<!-- benchmark-source-start -->\r?\n```julia\r?\n(.*?)\r?\n```\r?\n<!-- benchmark-source-end -->"

source = read(BENCHMARK_PAGE, String)
matched = match(BLOCK_PATTERN, source)
isnothing(matched) && error("could not find the julia_vs_r example in $BENCHMARK_PAGE")

if "--parse-only" in ARGS
    Meta.parse("begin\n$(matched.captures[1])\nend")
    println("benchmark source extracted and parsed")
else
    result = include_string(Main, matched.captures[1], BENCHMARK_PAGE)
    show(stdout, MIME"text/markdown"(), result)
    println()
end
