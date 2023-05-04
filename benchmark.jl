using WaterLily, StaticArrays, Plots
using LinearAlgebra: norm2


_nthread = Threads.nthreads()
if _nthread==1
    @warn "WaterLily.jl is running on a single thread.\n
Launch Julia with multiple threads to enable multithreaded capabilities:\n
    \$julia -t auto $PROGRAM_FILE"
else
    print("WaterLily.jl is running on ", _nthread, " thread(s)\n")
end

function circle(D=32;Re=100,U=1,ϵ=0.5)
    # Line segment SDF
    function sdf(x,t)
        return norm2(x) - D/2
    end
    # Oscillating motion and rotation
    function map(x,t)
        return x.-SVector(5D,4D)
    end
    Simulation((16D+2,8D+2),[U, 0.],D;U,ν=U*D/Re,body=AutoBody(sdf, map),ϵ)
end

function circleShape(h,k,r)
    θ = LinRange(0, 2π, 500)
    h .+ r*sin.(θ), k.+ r*cos.(θ)
end

nb_snapshot = 24*16
cycle = range(0, 24π, length=nb_snapshot)

D=32
Re=100
U=1
ϵ=0.5
swimmer = circle(D;Re,U,ϵ)

thrust = []
lift = []

# plot the vorcity ω=curl(u) scaled by the body length L and flow speed U
function plot_vorticity(sim,t)
	@inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
	contourf(sim.flow.f[:,:,1]',
			 color=palette(:BuGn), clims=(-2, 2), linewidth=0,
			 aspect_ratio=:equal, legend=true, border=:none)
    plot!(circleShape(5D,4D,D/2), seriestype=[:shape], lw=0.5,
    c=:blue, linecolor=:black,
    legend=false, fillalpha=1, aspect_ratio=1)
    append!(thrust,vec(sim.flow.f[:,:,1]))
    append!(lift,vec(sim.flow.f[:,:,2]))
end

# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=true)
	plot_vorticity(swimmer,t)
end

CSV.write("C:/Users/blagn771/Desktop/Thrust.csv", Tables.table(reshape(thrust, :, nb_snapshot)), writeheader=false)
CSV.write("C:/Users/blagn771/Desktop/Lift.csv", Tables.table(reshape(lift, :, nb_snapshot)), writeheader=false)

# function get_force(sim, t)
# 	sim_step!(sim, t, remeasure=true)
# 	return WaterLily.∮nds(sim.flow.p, sim.body, t*sim.L/sim.U) ./ (0.5*sim.L*sim.U^2)
# end
# forces = [get_force(swimmer, t) for t ∈ sim_time(swimmer) .+ cycle]

# scatter(cycle, [first.(forces), last.(forces)],
# 		labels=permutedims(["thrust", "side"]),
# 		xlabel="scaled time",
# 		ylabel="scaled force for Re=250, D=32")