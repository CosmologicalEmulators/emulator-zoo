using ArgParse

settings = ArgParseSettings()
@add_arg_table settings begin
    "--path-input", "-i"; required=true
    "--path-output", "-o"; required=true
    "--queue"; default="long"
    "--threads"; arg_type=Int; default=16
    "--memory-mb"; arg_type=Int; default=18000
end
arguments = parse_args(settings)
project_directory = @__DIR__
trainer = joinpath(project_directory, "trainer.jl")
for spectrum in ("Pmm", "Pcb")
    artifact = spectrum == "Pmm" ? "Pk_lin_mm" : "Pk_lin_cb"
    log_directory = joinpath(abspath(arguments["path-output"]), artifact)
    mkpath(log_directory)
    command = `bsub -q $(arguments["queue"])
        -o $(joinpath(log_directory, "job.out"))
        -e $(joinpath(log_directory, "job.err"))
        -n $(arguments["threads"])
        -M $(arguments["memory-mb"])
        -R span[hosts=1]
        julia -t $(arguments["threads"]) --project=$project_directory
        $trainer --spectrum $spectrum
        --path-input $(abspath(arguments["path-input"]))
        --path-output $(abspath(arguments["path-output"]))`
    println("Submitting Mapse trainer for spectrum=$spectrum")
    run(command)
end

