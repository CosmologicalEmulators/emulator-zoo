# This script preserves the ACE scientific calculation. The committed fixture
# is authoritative; do not rerun this script during tests.
using AbstractCosmologicalEmulators
using DataInterpolations
using FastGaussQuadrature
using Integrals
using OrdinaryDiffEqTsit5
using Printf
using PyCall
using SciMLSensitivity

const BACKGROUND_COSMOLOGY = Base.get_extension(
    AbstractCosmologicalEmulators,
    :BackgroundCosmologyExt,
)
BACKGROUND_COSMOLOGY === nothing && error("BackgroundCosmologyExt is unavailable")

input_lines = filter(
    line -> !startswith(line, "#") && !isempty(strip(line)),
    readlines(joinpath(@__DIR__, "reference_inputs.txt")),
)
values = parse.(Float64, split(strip(only(input_lines))))
names = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mν", "w0", "wa"]
p = Dict(names .=> values)
h = p["H0"] / 100
Omega_cb = (p["ombh2"] + p["omch2"]) / h^2
classy = pyimport("classy")
cosmology = classy.Class()
cosmology.set(Dict(
    "output" => "mPk", "P_k_max_h/Mpc" => 20.0, "z_pk" => "0.0,3.",
    "h" => h, "omega_b" => p["ombh2"], "omega_cdm" => p["omch2"],
    "ln10^{10}A_s" => p["ln10As"], "n_s" => p["ns"], "tau_reio" => 0.0568,
    "N_ur" => 2.033, "N_ncdm" => 1, "m_ncdm" => p["Mν"],
    "use_ppf" => "yes", "w0_fld" => p["w0"], "wa_fld" => p["wa"],
    "fluid_equation_of_state" => "CLP", "cs2_fld" => 1.0,
    "Omega_Lambda" => 0.0, "Omega_scf" => 0.0,
))
cosmology.compute()
D_z, f_z = BACKGROUND_COSMOLOGY.D_f_z(
    p["z"], Omega_cb, h;
    mν=p["Mν"], w0=p["w0"], wa=p["wa"],
)
common = [Float64(cosmology.sigma(8.0 / h, p["z"])), Float64(cosmology.rs_drag),
    Float64(cosmology.Hubble(p["z"])) * 299792.458, Float64(cosmology.comoving_distance(p["z"])), D_z, f_z]
for (name, output) in (
    ("result_sigma8_basis", vcat(p["ln10As"], common)),
    ("result_ln10As_basis", vcat(Float64(cosmology.sigma8), common)),
)
    print(name)
    for value in output
        @printf(" %.17g", value)
    end
    println()
end
cosmology.struct_cleanup()
cosmology.empty()
