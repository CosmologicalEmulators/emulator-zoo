using ArgParse

settings = ArgParseSettings()
@add_arg_table settings begin
    "--dataset"; required=true
    "--output"; required=true
    "--steps-per-session"; arg_type=Int; default=1000
    "--sessions-per-rate"; arg_type=Int; default=10
    "--batch-size"; arg_type=Int; default=512
end
arguments = parse_args(settings)

project = @__DIR__
trainer = joinpath(project, "trainer.jl")
julia = Base.julia_cmd()
for spectrum in ("TT", "TE", "EE", "PP")
    println("Training spectrum=$spectrum")
    env = copy(ENV)
    env["CAPSE_STEPS_PER_SESSION"] = string(arguments["steps-per-session"])
    env["CAPSE_SESSIONS_PER_RATE"] = string(arguments["sessions-per-rate"])
    env["CAPSE_BATCH_SIZE"] = string(arguments["batch-size"])
    run(setenv(`$julia --project=$project --startup-file=no $trainer
        --spectrum $spectrum -i $(abspath(arguments["dataset"]))
        -o $(abspath(arguments["output"]))`, env))
end
println("Trained TT, TE, EE, and PP")
