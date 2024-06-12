using LES_ML
using Documenter

DocMeta.setdocmeta!(LES_ML, :DocTestSetup, :(using LES_ML); recursive=true)

makedocs(;
    modules=[LES_ML],
    authors="tobyvg <tobyvangastelen@gmail.com> and contributors",
    sitename="LES_ML.jl",
    format=Documenter.HTML(;
        canonical="https://tobyvg.github.io/LES_ML.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tobyvg/LES_ML.jl",
    devbranch="main",
)
