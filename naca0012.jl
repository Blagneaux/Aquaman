using WaterLily, Plots, StaticArrays
using LinearAlgebra: norm2
using Interpolations
using Missings
using CSV
using Tables
using Statistics


fit = y -> scale(
    interpolate(y, BSpline(Quadratic(Line(OnGrid())))),
    range(0,1,length=length(y))
)

width = [0.02, 0.06, 0.06, 0.055, 0.04, 0.025, 0.01]
thkFish = fit(width)

# half thickness function
    function yt(x)
        x₁ = clamp(x[1]/C, 0., 1.)
        return (0.12/0.2) * (0.2969√x₁ - 0.1260*x₁ - 0.3516*x₁^2 + 0.2843*x₁^3 - 0.1036*x₁^4)
    end

function naca(C, Θ, St, Re, U)
    
    # Line segment SDF
    function sdfLine(x,t)
        thk=0.1 #1+√2
        y = x .- SVector(clamp(x[1],0.,C), 0.)
        √sum(abs2,y)-thk/2 - 1
    end

    # NACA sdf
    s(x) = clamp(x[1]/C, 0., 1.)
    sdfNaca(x,t) = √sum(abs2, x - C*SVector(s(x), 0.)) - C*yt(x)

    # Pulse from St
    ω = 2π*St*U/C

    function map(x,t)
        xc = x - [5C/4, 2C]
        α = Θ*sin(ω*t + π/2)
        R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
        xc = R*xc
        return xc + [C/4, 0.]
    end

    body = AutoBody(sdfNaca,map) + AutoBody(sdfLine,map)

    Simulation((8C+2,4C+2), [U, 0.], C; U, ν=U*C/Re, body=body)
end

nb_snapshot = 24*64
cycle = range(0, 48π, length=nb_snapshot) # 48π

C=64
Re=200
U=1
St=0.2
Θ=π/6
swimmer = naca(C,Θ,St,Re,U)

pos0 = [0., C*yt([0.,0.])]
pos1 = [C/4, C*yt([C/4,0])]
pos2 = [2C/4, C*yt([2C/4,0.])]
pos3 = [3C/4, C*yt([3C/4,0.])]
pos4 = [C/4, -C*yt([C/4,0])]
pos5 = [2C/4, -C*yt([2C/4,0.])]
pos6 = [3C/4, -C*yt([3C/4,0.])]

veloX = []; veloY = []; pressure = []
p1 = []; p2 = []; p3 = []; p4 = []; p5 = []; p6 = []; p7 = []; p8 = []; p9 = []; p10 = []; p11 = []
u1 = []; u2 = []; u3 = []; u4 = []; u5 = []; u6 = []; u7 = []; u8 = []; u9 = []; u10 = []; u11 = []
v1 = []; v2 = []; v3 = []; v4 = []; v5 = []; v6 = []; v7 = []; v8 = []; v9 = []; v10 = []; v11 = []
p₀ = []; p₁ = []; p₂ = []; p₃ = []; p₄ = []; p₅ = []; p₆ = []

# plot the vorcity ω=curl(u) scaled by the body length L and flow speed U
function plot_vorticity(sim,t)

    ω = 2π*St*U/C

    function map2(x,t)
        xc = x - [C/4, 0]
        α = (π/6)*sin(ω*t + π/2)
        R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
        xc = R*xc
        return xc + [5C/4, 2C]
    end
    
    posR0 = Int.(round.(map2(pos0,t* sim.L / sim.U)))
    posR1 = Int.(round.(map2(pos1,t* sim.L / sim.U)))
    posR2 = Int.(round.(map2(pos2,t* sim.L / sim.U)))
    posR3 = Int.(round.(map2(pos3,t* sim.L / sim.U)))
    posR4 = Int.(round.(map2(pos4,t* sim.L / sim.U)))
    posR5 = Int.(round.(map2(pos5,t* sim.L / sim.U)))
    posR6 = Int.(round.(map2(pos6,t* sim.L / sim.U)))

    # append!(veloX,vec(sim.flow.u[:,:,1]))
    # append!(veloY,vec(sim.flow.u[:,:,2]))
    # append!(pressure,vec(sim.flow.p))

    # append!(p1,[sim.flow.p[4C,Int(1.75C)]-sim.flow.p[3,2C+1]]); append!(u1,[sim.flow.u[4C,Int(1.75C),1]]); append!(v1,[sim.flow.u[4C,Int(1.75C),2]])
    # append!(p2,[sim.flow.p[4C,2C]-sim.flow.p[3,2C+1]]); append!(u2,[sim.flow.u[4C,2C,1]]); append!(v2,[sim.flow.u[4C,2C,2]])
    # append!(p3,[sim.flow.p[4C,Int(2.25C)]-sim.flow.p[3,2C+1]]); append!(u3,[sim.flow.u[4C,Int(2.25C),1]]); append!(v3,[sim.flow.u[4C,Int(2.25C),2]])
    # append!(p4,[sim.flow.p[4C,Int(2.5C)]-sim.flow.p[3,2C+1]]); append!(u4,[sim.flow.u[4C,Int(2.5C),1]]); append!(v4,[sim.flow.u[4C,Int(2.5C),2]])
    # append!(p5,[sim.flow.p[4C,Int(2.75C)]-sim.flow.p[3,2C+1]]); append!(u5,[sim.flow.u[4C,Int(2.75C),1]]); append!(v5,[sim.flow.u[4C,Int(2.75C),2]])
    # append!(p6,[sim.flow.p[4C,2C]-sim.flow.p[3,2C+1]]); append!(u6,[sim.flow.u[4C,2C,1]]); append!(v6,[sim.flow.u[4C,2C,2]])
    # append!(p7,[sim.flow.p[4C,Int(2.25C)]-sim.flow.p[3,2C+1]]); append!(u7,[sim.flow.u[4C,Int(2.25C),1]]); append!(v7,[sim.flow.u[4C,Int(2.25C),2]])
    # append!(p8,[sim.flow.p[4C,Int(2.5C)]-sim.flow.p[3,2C+1]]); append!(u8,[sim.flow.u[4C,Int(2.5C),1]]); append!(v8,[sim.flow.u[4C,Int(2.5C),2]])
    # append!(p9,[sim.flow.p[4C,Int(2.75C)]-sim.flow.p[3,2C+1]]); append!(u9,[sim.flow.u[4C,Int(2.75C),1]]); append!(v9,[sim.flow.u[4C,Int(2.75C),2]])
    # append!(p10,[sim.flow.p[4C,3C]-sim.flow.p[3,2C+1]]); append!(u10,[sim.flow.u[4C,3C,1]]); append!(v10,[sim.flow.u[4C,3C,2]])
    # append!(p11,[sim.flow.p[4C,Int(3.25C)]-sim.flow.p[3,2C+1]]); append!(u11,[sim.flow.u[4C,Int(3.25C),1]]); append!(v11,[sim.flow.u[4C,Int(3.25C),2]])

    coord0 = [[posR0[1]-1,4C+2-posR0[2]-1],[posR0[1]-1,4C+2-posR0[2]],[posR0[1]-1,4C+2-posR0[2]+1],[posR0[1],4C+2-posR0[2]-1],[posR0[1],4C+2-posR0[2]],[posR0[1],4C+2-posR0[2]+1],[posR0[1]+1,4C+2-posR0[2]-1],[posR0[1]+1,4C+2-posR0[2]],[posR0[1]+1,4C+2-posR0[2]+1],[posR0[1],4C+2-posR0[2]+2],[posR0[1],4C+2-posR0[2]-2]]
    coord1 = [[posR1[1]-1,4C+2-posR1[2]-1],[posR1[1]-1,4C+2-posR1[2]],[posR1[1]-1,4C+2-posR1[2]+1],[posR1[1],4C+2-posR1[2]-1],[posR1[1],4C+2-posR1[2]],[posR1[1],4C+2-posR1[2]+1],[posR1[1]+1,4C+2-posR1[2]-1],[posR1[1]+1,4C+2-posR1[2]],[posR1[1]+1,4C+2-posR1[2]+1],[posR1[1],4C+2-posR1[2]+2],[posR1[1],4C+2-posR1[2]-2]]
    coord2 = [[posR2[1]-1,4C+2-posR2[2]-1],[posR2[1]-1,4C+2-posR2[2]],[posR2[1]-1,4C+2-posR2[2]+1],[posR2[1],4C+2-posR2[2]-1],[posR2[1],4C+2-posR2[2]],[posR2[1],4C+2-posR2[2]+1],[posR2[1]+1,4C+2-posR2[2]-1],[posR2[1]+1,4C+2-posR2[2]],[posR2[1]+1,4C+2-posR2[2]+1],[posR2[1],4C+2-posR2[2]+2],[posR2[1],4C+2-posR2[2]-2]]
    coord3 = [[posR3[1]-1,4C+2-posR3[2]-1],[posR3[1]-1,4C+2-posR3[2]],[posR3[1]-1,4C+2-posR3[2]+1],[posR3[1],4C+2-posR3[2]-1],[posR3[1],4C+2-posR3[2]],[posR3[1],4C+2-posR3[2]+1],[posR3[1]+1,4C+2-posR3[2]-1],[posR3[1]+1,4C+2-posR3[2]],[posR3[1]+1,4C+2-posR3[2]+1],[posR3[1],4C+2-posR3[2]+2],[posR3[1],4C+2-posR3[2]-2]]
    coord4 = [[posR4[1]-1,4C+2-posR4[2]-2],[posR4[1]-1,4C+2-posR4[2]-1],[posR4[1]-1,4C+2-posR4[2]],[posR4[1],4C+2-posR4[2]-2],[posR4[1],4C+2-posR4[2]-1],[posR4[1],4C+2-posR4[2]],[posR4[1]+1,4C+2-posR4[2]-2],[posR4[1]+1,4C+2-posR4[2]-1],[posR4[1]+1,4C+2-posR4[2]],[posR4[1],4C+2-posR4[2]+1],[posR4[1],4C+2-posR4[2]-3]]
    coord5 = [[posR5[1]-1,4C+2-posR5[2]-2],[posR5[1]-1,4C+2-posR5[2]-1],[posR5[1]-1,4C+2-posR5[2]],[posR5[1],4C+2-posR5[2]-2],[posR5[1],4C+2-posR5[2]-1],[posR5[1],4C+2-posR5[2]],[posR5[1]+1,4C+2-posR5[2]-2],[posR5[1]+1,4C+2-posR5[2]-1],[posR5[1]+1,4C+2-posR5[2]],[posR5[1],4C+2-posR5[2]+1],[posR5[1],4C+2-posR5[2]-3]]
    coord6 = [[posR6[1]-1,4C+2-posR6[2]-2],[posR6[1]-1,4C+2-posR6[2]-1],[posR6[1]-1,4C+2-posR6[2]],[posR6[1],4C+2-posR6[2]-2],[posR6[1],4C+2-posR6[2]-1],[posR6[1],4C+2-posR6[2]],[posR6[1]+1,4C+2-posR6[2]-2],[posR6[1]+1,4C+2-posR6[2]-1],[posR6[1]+1,4C+2-posR6[2]],[posR6[1],4C+2-posR6[2]+1],[posR6[1],4C+2-posR6[2]-3]]
  
    mean0 = 0; mean1 = 0; mean2 = 0; mean3 = 0; mean4 = 0; mean5 = 0; mean6 = 0
    count0 = 0; count1 = 0; count2 = 0; count3 = 0; count4 = 0; count5 = 0; count6 = 0

    for i ∈ range(1, length(coord0))
        (sim.flow.μ₀[coord0[i][1],coord0[i][2],1] ≤ 0.5) && (mean0 += sim.flow.p[coord0[i][1],coord0[i][2]];count0 += 1)
        (sim.flow.μ₀[coord1[i][1],coord1[i][2],1] ≤ 0.5) && (mean1 += sim.flow.p[coord1[i][1],coord1[i][2]];count1 += 1)
        (sim.flow.μ₀[coord2[i][1],coord2[i][2],1] ≤ 0.5) && (mean2 += sim.flow.p[coord2[i][1],coord2[i][2]];count2 += 1)
        (sim.flow.μ₀[coord3[i][1],coord3[i][2],1] ≤ 0.5) && (mean3 += sim.flow.p[coord3[i][1],coord3[i][2]];count3 += 1)
        (sim.flow.μ₀[coord4[i][1],coord4[i][2],1] ≤ 0.5) && (mean4 += sim.flow.p[coord4[i][1],coord4[i][2]];count4 += 1)
        (sim.flow.μ₀[coord5[i][1],coord5[i][2],1] ≤ 0.5) && (mean5 += sim.flow.p[coord5[i][1],coord5[i][2]];count5 += 1)
        (sim.flow.μ₀[coord6[i][1],coord6[i][2],1] ≤ 0.5) && (mean6 += sim.flow.p[coord6[i][1],coord6[i][2]];count6 += 1)
    end#

    # 7 points on the naca
    append!(p₀,[mean0/count0 - sim.flow.p[3,2C+1]])
    append!(p₁,[mean1/count1 - sim.flow.p[3,2C+1]])
    append!(p₂,[mean2/count2 - sim.flow.p[3,2C+1]])
    append!(p₃,[mean3/count3 - sim.flow.p[3,2C+1]])
    append!(p₄,[mean4/count4 - sim.flow.p[3,2C+1]])
    append!(p₅,[mean5/count5 - sim.flow.p[3,2C+1]])
    append!(p₆,[mean6/count6 - sim.flow.p[3,2C+1]])
    
    # s = zeros(Int64, size(sim.flow.p))
    # for i ∈ range(1, length(coord0))
    #     if sim.flow.μ₀[coord0[i][1],coord0[i][2],1] ≤ 0.5
    #         s[coord0[i][1],coord0[i][2]] = -9
    #     end
    #     if sim.flow.μ₀[coord1[i][1],coord1[i][2],1] ≤ 0.5
    #         s[coord1[i][1],coord1[i][2]] = -9
    #     end
    #     if sim.flow.μ₀[coord2[i][1],coord2[i][2],1] ≤ 0.5
    #         s[coord2[i][1],coord2[i][2]] = -9
    #     end
    #     if sim.flow.μ₀[coord3[i][1],coord3[i][2],1] ≤ 0.5
    #         s[coord3[i][1],coord3[i][2]] = -9
    #     end
    #     if sim.flow.μ₀[coord4[i][1],coord4[i][2],1] ≤ 0.5
    #         s[coord4[i][1],coord4[i][2]] = -9
    #     end
    #     if sim.flow.μ₀[coord5[i][1],coord5[i][2],1] ≤ 0.5
    #         s[coord5[i][1],coord5[i][2]] = -9
    #     end
    #     if sim.flow.μ₀[coord6[i][1],coord6[i][2],1] ≤ 0.5
    #         s[coord6[i][1],coord6[i][2]] = -9
    #     end
    # end

	# @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
	# contourf(s',
    #         color=palette([:blue,:lightgrey,:red],9), clims=(-9, 9), linewidth=0,
	# 		 aspect_ratio=:equal, legend=true, border=:none)
    # contour!(swimmer.flow.μ₀[:,:,1]',
    #         aspect_ratio=:equal, legend=true, border=:none)
end

# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=true)
	plot_vorticity(swimmer,t)
end

plot(range(1,nb_snapshot), [p₀,p₁,p₂,p₃,p₄,p₅,p₆] , xlabel="x along the chord", ylabel="Cp for C=128, Re=200", 
label=permutedims(["on the nose", "at C/4 top", "at C/2 top", "at 3C/4 top", "at C/4 bot", "at C/2 bot", "at 3C/4 bot"]))

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

    CSV.write("C:/Users/blagn771/Desktop/pNoze.csv", Tables.table(p₀), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/p25Top.csv", Tables.table(p₁), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/p50Top.csv", Tables.table(p₂), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/p75Top.csv", Tables.table(p₃), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/p25Bot.csv", Tables.table(p₄), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/p50Bot.csv", Tables.table(p₅), writeheader=false)
    CSV.write("C:/Users/blagn771/Desktop/p75Bot.csv", Tables.table(p₆), writeheader=false)
end

# # Zeroth Moment map
# @gif for t ∈ cycle
#     measure!(swimmer, t*swimmer.L/swimmer.U)
#     contour(swimmer.flow.μ₀[:,:,1]',
#             aspect_ratio=:equal, legend=true, border=:none)
# end

# # Drag & Lift plot
# function get_force(sim, t)
# 	sim_step!(sim, t, remeasure=true, verbose=true)
# 	return WaterLily.∮nds(sim.flow.p .- sim.flow.p[3,2C+1], sim.body, t*sim.L/sim.U) ./ (0.5*sim.L*sim.U^2)
# end
# forces = [get_force(swimmer, t) for t ∈ sim_time(swimmer) .+ cycle]

# scatter(cycle, [first.(forces), last.(forces)],
# 		labels=permutedims(["thrust", "side"]),
# 		xlabel="scaled time",
# 		ylabel="scaled force for Re=200, C=64")

# # Cp plot
# function pressureCoef(sim, t)
#     s = missings(Int64, size(sim.flow.p))
#     for I ∈ inside(sim.flow.p)
#         x = loc(0,I)
#         d = sim.body.sdf(x,t)::Float64
#         abs(d) ≤ 1 && 2C ≤ x[2] && (s[I] = 1)
#     end

#     Cₚ = []
#     for i ∈ range(1,size(sim.flow.p)[1])
#         append!(Cₚ,[x for x ∈ skipmissing(s[i,:].*sim.flow.p[i,:])])
#     end

#     append!(Cₚs,[Cₚ .- sim.flow.p[3,2C+1]] )
# end

# Cₚs = []

# # make a gif over a swimming cycle
# @gif for t ∈ sim_time(swimmer) .+ cycle
# 	sim_step!(swimmer, t, remeasure=true, verbose=true)
#     pressureCoef(swimmer,t)
# end

# p₀ = []
# p₁ = []
# p₂ = []
# p₃ = []
# for i ∈ range(1, nb_snapshot)
#     append!(p₀, Cₚs[i][1])
#     append!(p₁, Cₚs[i][Int(round(length(Cₚs[i])/4))])
#     append!(p₂, Cₚs[i][Int(round(length(Cₚs[i])/2))])
#     append!(p₃, Cₚs[i][Int(round(3*length(Cₚs[i])/4))])
# end
# plot!(range(1,nb_snapshot), [p₀,p₁,p₂,p₃] , xlabel="x along the chord", ylabel="Cp for C=64, Re=200", label=permutedims(["on the nose", "at C/4", "at C/2", "at 3C/4"]))