using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2

using CSV, Tables
using Missings
using Statistics


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

nb_snapshot = 24*64
cycle = range(0, 48π, length=nb_snapshot)

D=32
Re=100
U=1
ϵ=0.5
swimmer = circle(D;Re,U,ϵ)

veloX = []
veloY = []
pressure = []

p1 = []; p2 = []; p3 = []; p4 = []; p5 = []; p6 = []; p7 = []; p8 = []; p9 = []; p10 = []; p11 = []
u1 = []; u2 = []; u3 = []; u4 = []; u5 = []; u6 = []; u7 = []; u8 = []; u9 = []; u10 = []; u11 = []
v1 = []; v2 = []; v3 = []; v4 = []; v5 = []; v6 = []; v7 = []; v8 = []; v9 = []; v10 = []; v11 = []

# plot the vorcity ω=curl(u) scaled by the body length L and flow speed U
function plot_vorticity(sim,t)
	@inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
	contourf(sim.flow.σ',
			 color=palette(:BuGn), clims=(-10, 10), linewidth=0,
			 aspect_ratio=:equal, legend=true, border=:none)
    plot!(circleShape(5D,4D,D/2), seriestype=[:shape], lw=0.5,
    c=:blue, linecolor=:black,
    legend=false, fillalpha=1, aspect_ratio=1)
    append!(veloX,vec(sim.flow.u[:,:,1]))
    append!(veloY,vec(sim.flow.u[:,:,2]))
    append!(pressure,vec(sim.flow.p))

    append!(p1,[sim.flow.p[8D,Int(1.5D)]]); append!(u1,[sim.flow.u[8D,Int(1.5D),1]]); append!(v1,[sim.flow.u[8D,Int(1.5D),2]])
    append!(p2,[sim.flow.p[8D,2D]]); append!(u2,[sim.flow.u[8D,2D,1]]); append!(v2,[sim.flow.u[8D,2D,2]])
    append!(p3,[sim.flow.p[8D,Int(2.5D)]]); append!(u3,[sim.flow.u[8D,Int(2.5D),1]]); append!(v3,[sim.flow.u[8D,Int(2.5D),2]])
    append!(p4,[sim.flow.p[8D,3D]]); append!(u4,[sim.flow.u[8D,3D,1]]); append!(v4,[sim.flow.u[8D,3D,2]])
    append!(p5,[sim.flow.p[8D,Int(3.5D)]]); append!(u5,[sim.flow.u[8D,Int(3.5D),1]]); append!(v5,[sim.flow.u[8D,Int(3.5D),2]])
    append!(p6,[sim.flow.p[8D,4D]]); append!(u6,[sim.flow.u[8D,4D,1]]); append!(v6,[sim.flow.u[8D,4D,2]])
    append!(p7,[sim.flow.p[8D,Int(4.5D)]]); append!(u7,[sim.flow.u[8D,Int(4.5D),1]]); append!(v7,[sim.flow.u[8D,Int(4.5D),2]])
    append!(p8,[sim.flow.p[8D,5D]]); append!(u8,[sim.flow.u[8D,5D,1]]); append!(v8,[sim.flow.u[8D,5D,2]])
    append!(p9,[sim.flow.p[8D,Int(5.5D)]]); append!(u9,[sim.flow.u[8D,Int(5.5D),1]]); append!(v9,[sim.flow.u[8D,Int(5.5D),2]])
    append!(p10,[sim.flow.p[8D,6D]]); append!(u10,[sim.flow.u[8D,6D,1]]); append!(v10,[sim.flow.u[8D,6D,2]])
    append!(p11,[sim.flow.p[8D,Int(6.5D)]]); append!(u11,[sim.flow.u[8D,Int(6.5D),1]]); append!(v11,[sim.flow.u[8D,Int(6.5D),2]])
end

function pressureCoef(sim, t)
    s = missings(Int64, size(sim.flow.p))
    for I ∈ inside(sim.flow.p)
        x = loc(0,I)
        d = sim.body.sdf(x,t)::Float64
        abs(d) ≤ 1 && x[2] ≤ 4D && (s[I] = 1)
    end

    Cₚ = []
    for i ∈ range(1,size(sim.flow.p)[1])
        append!(Cₚ,[x for x ∈ skipmissing(s[i,:].*sim.flow.p[i,:])])
    end

    append!(Cₚs,[Cₚ])
end

Cₚs = []

# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=true)
	# plot_vorticity(swimmer,t)
    pressureCoef(swimmer,t)
end

meanCₚ = zeros(length(Cₚs[1]))
for i ∈ range(1,length(Cₚs[1]))
    for j ∈ range(1, nb_snapshot)
        meanCₚ[i] += Cₚs[j][i]
    end
end
meanCₚ ./= length(Cₚs[1])

# CSV.write("C:/Users/blagn771/Desktop/MeanPressureCoefAroundCircle.csv", Tables.table(Cₚs), writeheader=false)


save = false

# save velocity fields, pressure field, Δt, for the whole simulation
# save pressure coefficient and velocity at some point behind the cylindre
if save
    CSV.write("C:/Users/blagn771/Desktop/VelocityX.csv", Tables.table(reshape(veloX, :, nb_snapshot)), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/velocityY.csv", Tables.table(reshape(veloY, :, nb_snapshot)), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/Pressure.csv", Tables.table(reshape(pressure, :, nb_snapshot)), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/DeltaT.csv", Tables.table(swimmer.flow.Δt), writeheader=false)

    CSV.write("C:/Users/blagn771/Desktop/p1_5D.csv", Tables.table(p1), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/u1_5D.csv", Tables.table(u1), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/v1_5D.csv", Tables.table(v1), writeheader=false)

    CSV.write("C:/Users/blagn771/Desktop/p2D.csv", Tables.table(p2), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/u2D.csv", Tables.table(u2), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/v2D.csv", Tables.table(v2), writeheader=false)

    CSV.write("C:/Users/blagn771/Desktop/p2_5D.csv", Tables.table(p3), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/u2_5D.csv", Tables.table(u3), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/v2_5D.csv", Tables.table(v3), writeheader=false)

    CSV.write("C:/Users/blagn771/Desktop/p3D.csv", Tables.table(p4), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/u3D.csv", Tables.table(u4), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/v3D.csv", Tables.table(v4), writeheader=false)

    CSV.write("C:/Users/blagn771/Desktop/p3_5D.csv", Tables.table(p5), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/u3_5D.csv", Tables.table(u5), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/v3_5D.csv", Tables.table(v5), writeheader=false)

    CSV.write("C:/Users/blagn771/Desktop/p4D.csv", Tables.table(p6), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/u4D.csv", Tables.table(u6), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/v4D.csv", Tables.table(v6), writeheader=false)

    CSV.write("C:/Users/blagn771/Desktop/p4_5D.csv", Tables.table(p7), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/u4_5D.csv", Tables.table(u7), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/v4_5D.csv", Tables.table(v7), writeheader=false)

    CSV.write("C:/Users/blagn771/Desktop/p5D.csv", Tables.table(p8), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/u5D.csv", Tables.table(u8), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/v5D.csv", Tables.table(v8), writeheader=false)

    CSV.write("C:/Users/blagn771/Desktop/p5_5D.csv", Tables.table(p9), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/u5_5D.csv", Tables.table(u9), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/v5_5D.csv", Tables.table(v9), writeheader=false)

    CSV.write("C:/Users/blagn771/Desktop/p6D.csv", Tables.table(p10), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/u6D.csv", Tables.table(u10), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/v6D.csv", Tables.table(v10), writeheader=false)

    CSV.write("C:/Users/blagn771/Desktop/p6_5D.csv", Tables.table(p11), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/u6_5D.csv", Tables.table(u11), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/v6_5D.csv", Tables.table(v11), writeheader=false)
end

# function get_force(sim, t)
# 	sim_step!(sim, t, remeasure=true)
# 	return WaterLily.∮nds(sim.flow.p, sim.body, t*sim.L/sim.U) ./ (0.5*sim.L*sim.U^2)
# end
# forces = [get_force(swimmer, t) for t ∈ sim_time(swimmer) .+ cycle]

# scatter(cycle, [first.(forces), last.(forces)],
# 		labels=permutedims(["thrust", "side"]),
# 		xlabel="scaled time",
# 		ylabel="scaled force for Re=250, D=16")

# CSV.write("C:/Users/blagn771/Desktop/Lift.csv", Tables.table(last.(forces)), writeheader=false)
# CSV.write("C:/Users/blagn771/Desktop/Drag.csv", Tables.table(first.(forces)), writeheader=false)

# plot(swimmer.flow.Δt, xlabel="solver iterations", ylabel="Δt", labels=permutedims(["Δt for Re=250, U=1, D=16"]))

# scatter(cycle, [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11],
#         labels=permutedims(["Pressure at 8D,1.5D", "at 8D,2D", "at 8D,2.5D","at 8D,3D","at 8D,3.5D","at 8D,4D","at 8D,4.5D", "at 8D,5D", "at 8D,5.5D", "at 8D,6D", "at 8D,6.5D"]),
#         xlabel="scaled time",
#         ylabel="Pressure coefficient at different points, for Re=250 and D=32")

plot!(range(1,180,length(Cₚs[1])), meanCₚ, xlabel="deg around the circle", ylabel="mean Cₚ", labels=permutedims(["Re=100 D=32"]))