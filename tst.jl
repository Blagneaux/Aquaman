using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images, Statistics
using LinearAlgebra: norm2
using ImageView

_nthread = Threads.nthreads()
if _nthread==1
    @warn "WaterLily.jl is running on a single thread.\n
Launch Julia with multiple threads to enable multithreaded capabilities:\n
    \$julia -t auto $PROGRAM_FILE"
else
    print("WaterLily.jl is running on ", _nthread, " thread(s)\n")
end

function block(L=2^5;Re=250,U=1,amp=π/4,ϵ=0.5,thk=2ϵ+√2)
    # Line segment SDF
    function sdf(x,t)
        y = x .- SVector(0.,clamp(x[2],-L/2,L/2))
        √sum(abs2,y)-thk/2
    end
    # Oscillating motion and rotation
    function map(x,t)
        α = amp*cos(t*U/L); R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
        R * (x.-SVector(3L,4L))
    end

    line = AutoBody(sdf, map)

    # Counter-Oscillating motion and rotation
    function map2(x,t)
        α = -amp*cos(t*U/L); R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
        R * (x.-SVector(3L,4L))
    end

    # Circle SDF
    center, radius = [L,3.5L], L/4
    sdfC(x,t) = norm2(x .- center) - radius

    # Translation of the circle
    mapC(x,t) = x - SA[U*t/4,0]

    circle = AutoBody(sdf, map2)

    Simulation((6L+2,6L+2),zeros(2),L;U,ν=U*L/Re,body=WaterLily.intersectBodies(line,circle),ϵ)
end

function double(radius=8;Re=250,U=1,amp=π/4,ϵ=0.5,thk=2ϵ+√2)
    circle1(x,t) = √(x'*x) - 1.5radius
    circle2(x,t) = √(x'*x) - 1radius
    circle3(x,t) = √(x'*x) - 1.125radius
    map1(x,t) = x-SA[3radius,3radius]
    map2(x,t) = x-SA[3radius,3radius]

    # Line segment SDF
    L=radius
    function sdf(x,t)
        y = x .- SVector(0.,clamp(x[2],-L/2,L/2))
        √sum(abs2,y)-thk/2
    end
    # Oscillating motion and rotation
    function map(x,t)
        α = amp*cos(t*U/L); R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
        R * (x.-SVector(3L,3L))
    end

    line = AutoBody(sdf, map)

    body = WaterLily.morphBodies(AutoBody(circle1,map1), line,(a,b,t)->a+(b-a)*(sin(0.1t)*0.5+0.5)) 
    Simulation((6radius+2,6radius+2), zeros(2), radius; U, ν=U*radius/Re, body)
end

L=2^5
swimmer = double()
cycle = range(0, 8π, length=24*8)

moyFull = []

foreach(rm, readdir("C:/Users/blagn771/Desktop/PseudoGif", join=true))

# function computeSDF(sim, t)
#     s = copy(sim.flow.p)
#     for I ∈ inside(s)
#         x = loc(0, I)
#         s[I] = sim.body.sdf(x,t*sim.L/sim.U)::Float64
#     end

#     contourf(s', clims=(-L/2,L), linewidth=0,
#             aspect_ratio=:equal, legend=true, border=:none)
#     # contour(swimmer.flow.μ₀[:,:,1]',
#     #         aspect_ratio=:equal, legend=true, border=:none)
#     savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
# end

# @gif for t ∈ sim_time(swimmer) .+ cycle
#     sim_step!(swimmer, t, remeasure=true, verbose=true)
#     computeSDF(swimmer, t)
# end

# plot the vorcity ω=curl(u) scaled by the body length L and flow speed U
function plot_vorticity(sim,t)
	@inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
	contourf(sim.flow.μ₀[:,:,1]',
			 color=palette(:BuGn), clims=(-1, 1), linewidth=0,
			 aspect_ratio=:equal, legend=true, border=:none)
    savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
end


# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=false)
	plot_vorticity(swimmer,t)
end

# scatter([i for i in range(1,length(moyFull))], [moyFull],
#     labels=permutedims(["Mean pressure coefficient on the whole window"]),
#     xlabel="scaled time",
#     ylabel="scaled pressure")