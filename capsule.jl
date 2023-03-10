using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2

fit = y -> scale(
        interpolate(y, BSpline(Quadratic(Line(OnGrid())))),
        range(0,1,length=length(y))
    )

width = [0.14, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04]
thk = fit(width)

# envelope = [0, 0, 0.2, 0.4, 0.6, 0.8, 1.]
# amp = fit(envelope)

@fastmath kernᵦ(d) = 0.5+0.5d+0.5sin(π*d)/π
μᵦ(d,ϵ) = kernᵦ(clamp(d/ϵ,-1,1))

function capsule(L, St, A)
	# fraction along fish length
	s(x) = clamp(x[1]/L, 0, 1)

	# fish geometry: thickened line SDF
	sdf(x,t) = √sum(abs2, x - L * SVector(s(x), 0.)) - L * thk(s(x))

    # fish motion: travelling wave
	U = 0.89
	ω = 2π * St * U/(2A * L)

    stop = 2A/St*4.8*23/24
    
    function map(x,t)
        while t < stop*L
            xc = x - [517 + 9.7,258/2] + [U*t,0.]
            amp = 25π/180
            # xc = x - [517,258/2] + [U*t,0.]
            # return xc - SVector(0., 25 * amp(s(xc)) * sin(1.5s(xc)-ω*t)) 
            α = amp*sin(4t*U/L)
            R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
            I₂ = @SMatrix [1 0; 0 1]
            # xc = R*xc
            xc = (μᵦ(xc[1],10).*R + (1-μᵦ(xc[1],10)).*I₂)*xc
            return xc + [9.7,0.]
        end
        xc = x - [517 + 9.7,258/2]+ [U*stop*L,0.]
        amp = 25π/180
        α = amp*sin(4*stop*L*U/L)
        R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
        I₂ = @SMatrix [1 0; 0 1]
        xc = (μᵦ(xc[1],10).*R + (1-μᵦ(xc[1],10)).*I₂)*xc
        return xc + [9.7,0.]
    end

	# make the simulation
	return SVector(sdf,map)
end