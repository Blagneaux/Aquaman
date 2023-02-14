using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2

fit = y -> scale(
        interpolate(y, BSpline(Quadratic(Line(OnGrid())))),
        range(0,1,length=length(y))
    )

width = [0, 0.05, 0.07, 0.085, 0.0925, 0.0925, 0.0925, 0.0925, 0.0925, 0.0925, 0.0925, 0.09, 0.088, 0.086, 0.084, 0.082, 0.08, 0.078, 0.076, 0.074, 0.072, 0.07, 0.068, 0.066, 0.064, 0.062, 0.06, 0.058, 0.056, 0.054, 0.052, 0.05, 0.048, 0.046, 0.044, 0.042, 0.04, 0.038, 0.036, 0.034, 0.032, 0.03, 0.028, 0.026, 0.024, 0.022, 0.02, 0.018, 0.016, 0.014, 0.012, 0.01, 0.008, 0.006, 0.004, 0.002, 0]
thk = fit(width)

envelope = [0, 0, 0.2, 0.4, 0.6, 0.8, 1.]
amp = fit(envelope)

function capsule(L, St, A)
	# fraction along fish length
	s(x) = clamp(x[1]/L, 0, 1)

	# fish geometry: thickened line SDF
	sdf(x,t) = √sum(abs2, x - L * SVector(s(x), 0.)) - L * thk(s(x))

    # fish motion: travelling wave
	U = 0.89
	ω = 2π * St * U/(2A * L)
    
    function map(x,t)
        xc = x - [517 + 9.7 + 6.5,258/2] + [U*t,0.]
        amp = 25*π/180
        # return xc - SVector(0., 22.5 * amp(s(xc)) * sin(-ω*t)) 
        α = amp*sin(4t*U/L)
        R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
        if xc[1]>0
            xc = R*xc
        end
        return xc + [9.7+6.5,0.]
    end

	# make the simulation
	return SVector(sdf,map)
end