package main

import (
	"math"
	"testing"
)

func TestUTMMatchesPROJ(t *testing.T) {
	// EPSG:4326 -> EPSG:326zz, computed with PROJ.
	cases := []struct {
		epsg     int
		lon, lat float64
		e, n     float64
	}{
		{32634, 23.64, 46.77, 701562.8654, 5182990.1555},
		{32635, 26.02, 44.94, 422680.4253, 4976752.3003},
		{32635, 29.50, 45.30, 696006.7347, 5019317.9653},
		{32635, 24.10, 43.82, 266789.0680, 4855969.2155},
	}
	for _, c := range cases {
		u, _ := utmFromEPSG(c.epsg)
		e, n := u.forward(c.lon, c.lat)
		if math.Abs(e-c.e) > 1e-3 || math.Abs(n-c.n) > 1e-3 {
			t.Errorf("%v: got %.4f %.4f", c, e, n)
		}
		lon, lat := u.inverse(c.e, c.n)
		if math.Abs(lon-c.lon) > 1e-9 || math.Abs(lat-c.lat) > 1e-9 {
			t.Errorf("inverse %v: got %.10f %.10f", c, lon, lat)
		}
	}
}

func grid(size int, f func(r, c int) uint8) []uint8 {
	g := make([]uint8, size*size)
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			g[r*size+c] = f(r, c)
		}
	}
	return g
}

func TestClearCutIsFoundAndSliversAreNot(t *testing.T) {
	const n = 40
	before := grid(n, func(r, c int) uint8 { return tree })
	after := grid(n, func(r, c int) uint8 {
		switch {
		case r >= 5 && r < 15 && c >= 5 && c < 17: // a 10 x 12 clearing
			return grass
		case c == 30: // a one-pixel line: misregistration, not change
			return crop
		case r == 35 && c == 35: // a single pixel
			return bare
		}
		return tree
	})
	k := Kinds[0]
	lab, patches := Components(Open(Mask(before, after, k), n), n, 10)
	if len(patches) != 1 {
		t.Fatalf("want one patch, got %d: %+v", len(patches), patches)
	}
	p := patches[0]
	if p.Pixels != 120 || p.MinR != 5 || p.MaxR != 14 || p.MinC != 5 || p.MaxC != 16 {
		t.Errorf("patch %+v", p)
	}
	if lab[10*n+10] != 1 || lab[20*n+30] != 0 {
		t.Errorf("labels wrong")
	}
	// Nothing became built-up.
	if _, ps := Components(Open(Mask(before, after, Kinds[1]), n), n, 1); len(ps) != 0 {
		t.Errorf("construction found in a clear-cut: %+v", ps)
	}
}

func TestSmallPatchesAreDroppedAndLabelsStayDense(t *testing.T) {
	const n = 30
	m := make([]bool, n*n)
	set := func(r0, c0, h, w int) {
		for r := r0; r < r0+h; r++ {
			for c := c0; c < c0+w; c++ {
				m[r*n+c] = true
			}
		}
	}
	set(0, 0, 2, 2)   // 4 pixels: dropped
	set(10, 10, 5, 5) // 25: kept
	set(20, 0, 4, 4)  // 16: kept
	lab, ps := Components(m, n, 10)
	if len(ps) != 2 {
		t.Fatalf("got %d patches", len(ps))
	}
	if lab[0] != 0 || lab[10*n+10] != 1 || lab[20*n] != 2 {
		t.Errorf("labels %d %d %d", lab[0], lab[10*n+10], lab[20*n])
	}
}

func TestTransitionsSkipNoData(t *testing.T) {
	a := []uint8{tree, tree, crop, noData}
	b := []uint8{tree, built, built, built}
	m := Transitions(a, b)
	if m[tree][tree] != 1 || m[tree][built] != 1 || m[crop][built] != 1 {
		t.Errorf("%v", m)
	}
	total := 0
	for _, row := range m {
		for _, v := range row {
			total += v
		}
	}
	if total != 3 {
		t.Errorf("total %d", total)
	}
}

func TestConfirmNeedsAnIndexDrop(t *testing.T) {
	m := []bool{true, true, true, false}
	before := []int16{7000, 7000, math.MinInt16, 7000}
	after := []int16{5500, 6500, 2000, 1000}
	got := Confirm(m, before, after, 1000)
	want := []bool{true, false, false, false}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v want %v", got, want)
		}
	}
}
