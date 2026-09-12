const README_PATH = normpath(joinpath(@__DIR__, "..", "README.md"))
const HOMEPAGE_PATH = joinpath(@__DIR__, "src", "index.md")

const HOMEPAGE_FRONTMATTER = """````@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Copulas.jl
  text:
  tagline: A Distributions.jl-compliant copula package.
  image:
    src: logo.svg
    alt: Copulas.jl
  actions:
    - theme: brand
      text: Getting started
      link: /manual/intro
    - theme: alt
      text: View on Github
      link: https://github.com/lrnv/Copulas.jl
    - theme: alt
      text: Bestiary
      link: /bestiary/elliptical
---
````

<!-- This file is generated from README.md by docs/sync_homepage.jl. -->

"""

function homepage_from_readme(readme::AbstractString)
    readme_only = r"(?s)<!-- README-ONLY:START -->.*?<!-- README-ONLY:END -->\s*"
    length(collect(eachmatch(readme_only, readme))) == 1 ||
        error("README.md must contain exactly one README-ONLY block")
    body = replace(readme, readme_only => ""; count=1)

    marker = r"<!-- DOCUMENTER-EXAMPLE: ([A-Za-z0-9_.-]+) -->\r?\n```julia"
    nmarkers = length(collect(eachmatch(marker, body)))
    body = replace(body, marker => s"```@example \1")
    occursin("<!-- DOCUMENTER-EXAMPLE:", body) &&
        error("README.md contains a malformed DOCUMENTER-EXAMPLE marker")
    nmarkers > 0 || error("README.md must contain at least one Documenter example marker")

    return HOMEPAGE_FRONTMATTER * strip(body) * "\n"
end

function sync_homepage()
    generated = homepage_from_readme(read(README_PATH, String))
    current = isfile(HOMEPAGE_PATH) ? read(HOMEPAGE_PATH, String) : ""
    current == generated || write(HOMEPAGE_PATH, generated)
    return nothing
end

sync_homepage()
