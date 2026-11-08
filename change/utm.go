package main

import "math"

// UTM <-> WGS84 (transverse Mercator, Krüger's series to fourth order in n): the same formulas as
// the Rust tile server, accurate to well under a millimetre within a zone.

const (
	wgsA  = 6378137.0
	wgsF  = 1 / 298.257223563
	utmK0 = 0.9996
	utmE0 = 500000.0
)

type utm struct{ lon0, n0 float64 }

func utmFromEPSG(epsg int) (utm, bool) {
	switch {
	case epsg > 32600 && epsg <= 32660:
		return utm{float64(epsg-32600-1)*6 - 180 + 3, 0}, true
	case epsg > 32700 && epsg <= 32760:
		return utm{float64(epsg-32700-1)*6 - 180 + 3, 10000000}, true
	}
	return utm{}, false
}

type tmSeries struct {
	a                  float64
	alpha, beta, delta [4]float64
	c                  float64
}

var series = func() tmSeries {
	n := wgsF / (2 - wgsF)
	n2, n3, n4 := n*n, n*n*n, n*n*n*n
	return tmSeries{
		a:     wgsA / (1 + n) * (1 + n2/4 + n4/64),
		alpha: [4]float64{n/2 - 2*n2/3 + 5*n3/16 + 41*n4/180, 13*n2/48 - 3*n3/5 + 557*n4/1440, 61*n3/240 - 103*n4/140, 49561 * n4 / 161280},
		beta:  [4]float64{n/2 - 2*n2/3 + 37*n3/96 - n4/360, n2/48 + n3/15 - 437*n4/1440, 17*n3/480 - 37*n4/840, 4397 * n4 / 161280},
		delta: [4]float64{2*n - 2*n2/3 - 2*n3 + 116*n4/45, 7*n2/3 - 8*n3/5 - 227*n4/45, 56*n3/15 - 136*n4/35, 4279 * n4 / 630},
		c:     2 * math.Sqrt(n) / (1 + n),
	}
}()

func (u utm) forward(lon, lat float64) (e, n float64) {
	s := series
	phi := lat * math.Pi / 180
	lam := (lon - u.lon0) * math.Pi / 180
	t := math.Sinh(math.Atanh(math.Sin(phi)) - s.c*math.Atanh(s.c*math.Sin(phi)))
	xi := math.Atan2(t, math.Cos(lam))
	eta := math.Atanh(math.Sin(lam) / math.Sqrt(1+t*t))
	x, y := eta, xi
	for j := 0; j < 4; j++ {
		k := 2 * float64(j+1)
		x += s.alpha[j] * math.Cos(k*xi) * math.Sinh(k*eta)
		y += s.alpha[j] * math.Sin(k*xi) * math.Cosh(k*eta)
	}
	return utmE0 + utmK0*s.a*x, u.n0 + utmK0*s.a*y
}

func (u utm) inverse(e, n float64) (lon, lat float64) {
	s := series
	xi := (n - u.n0) / (utmK0 * s.a)
	eta := (e - utmE0) / (utmK0 * s.a)
	xp, ep := xi, eta
	for j := 0; j < 4; j++ {
		k := 2 * float64(j+1)
		xp -= s.beta[j] * math.Sin(k*xi) * math.Cosh(k*eta)
		ep -= s.beta[j] * math.Cos(k*xi) * math.Sinh(k*eta)
	}
	chi := math.Asin(math.Sin(xp) / math.Cosh(ep))
	phi := chi
	for j := 0; j < 4; j++ {
		phi += s.delta[j] * math.Sin(2*float64(j+1)*chi)
	}
	lam := math.Atan2(math.Sinh(ep), math.Cos(xp))
	return u.lon0 + lam*180/math.Pi, phi * 180 / math.Pi
}
