using ArgParse

settings = ArgParseSettings()
@add_arg_table settings begin
    "--path-input", "-i"; required=true
    "--path-output", "-o"; required=true
    "--queue"; default="long"
    "--threads"; arg_type=Int; default=8
    "--memory-mb"; arg_type=Int; default=12000
end
arguments = parse_args(settings)
project_directory = @__DIR__
trainer = joinpath(project_directory, "trainer.jl")

for basis in ("ln10As", "sigma8")
    log_directory = joinpath(abspath(arguments["path-output"]), basis)
    mkpath(log_directory)
    command = `bsub -q $(arguments["queue"])
        -o $(joinpath(log_directory, "job.out"))
        -e $(joinpath(log_directory, "job.err"))
        -n $(arguments["threads"])
        -M $(arguments["memory-mb"])
        -R span[hosts=1]
        julia -t $(arguments["threads"]) --project=$project_directory
        $trainer --basis $basis
        --path-input $(abspath(arguments["path-input"]))
        --path-output $(abspath(arguments["path-output"]))`
    println("Submitting ACE trainer for basis=$basis")
    run(command)
end

