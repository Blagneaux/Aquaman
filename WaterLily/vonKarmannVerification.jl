using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")

function circle(radius=20;Re=250,n=3*2^5,m=6)
    center, ν = 2n+1, radius/Re
    body = AutoBody((x,t)->norm2(x .- center) - radius)
    Simulation((10n+2,4n+2), [1.,0.], radius*2; ν, body)
end

swimmer = circle()
cycle = range(0, 8, length=24*8)

# run the simulation a few cycles (this takes few seconds)
sim_step!(swimmer, 150, remeasure=true)
sim_time(swimmer)


function get_force(sim, t)
	sim_step!(sim, t, remeasure=true)
	return WaterLily.∮nds(sim.flow.p, sim.body, t*sim.L/sim.U) ./ (0.5*sim.L*sim.U^2)
end
forces = [get_force(swimmer, t) for t ∈ sim_time(swimmer) .+ cycle]

scatter(cycle, [first.(forces), last.(forces)],
		labels=permutedims(["thrust", "side"]),
		xlabel="scaled time",
		ylabel="scaled force")