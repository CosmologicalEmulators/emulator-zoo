spectra = ["TT", "TE", "EE"]
input = "/farmdisk1/mbonici/capse_axiclass_1000"
output = "/farmdisk1/mbonici/trained_axicapse_1000"

for spectrum in spectra
    # Construct the bsub command
    folder_output = output
    bsub_command = `bsub -P c7 -q medium -o /home/mbonici/emulator-zoo/Capse.jl/axiclass/job.out \
                        -e /home/mbonici/emulator-zoo/Capse.jl/axiclass/job.err -n 4 -M 12000 \
                        -R'span[hosts=1] select[hname!=teo22 && hname!=infne01 && hname!=totem04 && hname!=totem07 && hname!=totem08 && hname!=geant15 && hname!=geant16 && hname!=aiace12 && hname!=aiace13 && hname!=aiace14 && hname!=aiace15 && hname!=aiace16 && hname!=aiace17]' \
                        julia -t 4 --project=/home/mbonici/emulator-zoo/Capse.jl/axiclass \
                        /home/mbonici/emulator-zoo/Capse.jl/axiclass/trainer.jl \
                        --spectrum $spectrum \
                        -i $input \
                        -o $folder_output`

    # Print the command for debugging (optional)
    println("Submitting job with --spectrum=$spectrum")

    # Run the command
    run(bsub_command)
end
