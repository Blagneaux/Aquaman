using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2

include("capsule.jl")

_nthread = Threads.nthreads()
if _nthread==1
    @warn "WaterLily.jl is running on a single thread.\n
Launch Julia with multiple threads to enable multithreaded capabilities:\n
    \$julia -t auto $PROGRAM_FILE"
else
    print("WaterLily.jl is running on ", _nthread, " thread(s)\n")
end

function wall(a, b)
    function sdf(x,t)
        xa = x-a
        ba = b-a
        h = clamp(dot(xa,ba)/dot(ba,ba), 0.0, 1.0)
        return norm2(xa - ba*h) - 10
    end

    function map(x,t)
        xc = x - [5L, 258/2]
        return xc
    end

    return SVector(sdf, map)
end

L,A,St,U = 71.2,0.466,0.61,0.89
capsuleShape = capsule(L, St, A)

wallShape1 = wall([-600,110], [400,110])
wallShape2 = wall([-600,-110], [400,-110])

swimmerBody = WaterLily.addBodies(AutoBody(capsuleShape[1], capsuleShape[2]),AutoBody(wallShape1[1], wallShape1[2]), AutoBody(wallShape2[1],wallShape2[2]))
swimmer = Simulation((642,258), [0.,0.], L, U=0.89; ν=U*L/6070, body=swimmerBody)


# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0, 23*period/24, length=24)

foreach(rm, readdir("C:/Users/blagn771/Desktop/PseudoGif", join=true))

function computeSDF(sim, t)
    s = copy(sim.flow.p)
    for I ∈ inside(s)
        x = loc(0, I)
        s[I] = sim.body.sdf(x,t*sim.L/sim.U)::Float64
    end

    contourf(s', clims=(-L,2L), linewidth=0,
            aspect_ratio=:equal, legend=true, border=:none)
    savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
end

@time @gif for t ∈ sim_time(swimmer) .+ cycle
    sim_step!(swimmer, t, remeasure=true, verbose=true)
    computeSDF(swimmer, t)
end

# @gif for t ∈ cycle
# 	measure!(swimmer, t*swimmer.L/swimmer.U)
# 	contour(swimmer.flow.μ₀[:,:,1]',
# 			aspect_ratio=:equal, legend=true, border=:none)
# end

#225.172887 seconds (1.59 G allocations: 59.436 GiB, 3.35% gc time, 0.46% compilation time)
#1764.266766 seconds (15.36 G allocations: 513.906 GiB, 3.79% gc time, 0.10% compilation time)