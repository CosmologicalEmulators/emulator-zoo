spectra = ["TT", "TE", "EE", "PP"]
input = "/farmdisk1/mbonici/capse_class_lcdm_101"

for component in spectra
    # Construct the bsub command
    bsub_command = `bsub -P c7 -q medium -o /home/mbonici/emulator-zoo/Capse.jl/class_lcdm/job.out \
                        -e /home/mbonici/emulator-zoo/Capse.jl/class_lcdm/job.err -n 4 -M 12000 \
                        -R'span[hosts=1] select[hname!=teo22 && hname!=infne01 && hname!=totem04 && hname!=totem07 && hname!=totem08 && hname!=geant15 && hname!=geant16 && hname!=aiace12 && hname!=aiace13 && hname!=aiace14 && hname!=aiace15 && hname!=aiace16 && hname!=aiace17]' \
                        julia -t 4 --project=/home/mbonici/emulator-zoo/Capse.jl/class_mnuw0walcdm\
                        /home/mbonici/emulator-zoo/Capse.jl/class_lcdm/trainer.jl \
                        --spectra $spectra \
                        -i $input \
                        -o $folder_output`

    # Print the command for debugging (optional)
    println("Submitting job with --spectra=$spectra")

    # Run the command
    run(bsub_command)
end
