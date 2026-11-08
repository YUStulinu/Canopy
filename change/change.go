package main

import "math"

// Change between two land-cover maps of the same grid: which pixels went from one class to
// another, cleaned of edge noise, grouped into connected patches.

const (
	tree     = 0
	shrub    = 1
	grass    = 2
	crop     = 3
	built    = 4
	bare     = 5
	water    = 7
	noData   = 255
	nClasses = 10
)

// A kind of change: which transitions count.
type Kind struct {
	Name   string
	Before func(c uint8) bool
	After  func(c uint8) bool
}

var Kinds = []Kind{
	{"deforestation",
		func(c uint8) bool { return c == tree },
		func(c uint8) bool { return c == shrub || c == grass || c == crop || c == bare }},
	{"construction",
		func(c uint8) bool { return c != built && c != noData },
		func(c uint8) bool { return c == built }},
}

// Transitions counts pixels by (class before, class after); pixels without data on either date
// are left out.
func Transitions(a, b []uint8) [nClasses][nClasses]int {
	var m [nClasses][nClasses]int
	for i := range a {
		if a[i] < nClasses && b[i] < nClasses {
			m[a[i]][b[i]]++
		}
	}
	return m
}

// Mask marks the pixels where kind k happened.
func Mask(a, b []uint8, k Kind) []bool {
	m := make([]bool, len(a))
	for i := range a {
		m[i] = a[i] != noData && b[i] != noData && k.Before(a[i]) && k.After(b[i])
	}
	return m
}

// Confirm keeps only the pixels of a mask where a spectral index dropped by at least minDrop
// between the two dates (indices x 10000; math.MinInt16 = no data, which never confirms).
// Used for deforestation with the normalised burn ratio: a class map can flip from "tree cover"
// to "grassland" because two composites differ in season or haze, but cutting a forest removes
// the canopy's near-infrared signal and exposes soil and slash, so NBR falls.
func Confirm(m []bool, before, after []int16, minDrop int) []bool {
	out := make([]bool, len(m))
	for i := range m {
		if m[i] && before[i] != math.MinInt16 && after[i] != math.MinInt16 {
			out[i] = int(before[i])-int(after[i]) >= minDrop
		}
	}
	return out
}

// Open removes features thinner than three pixels (erosion then dilation with a 3 x 3 square):
// most one-pixel slivers along field and forest edges are misregistration or mixed pixels, not
// change.
func Open(m []bool, size int) []bool {
	return morph(morph(m, size, true), size, false)
}

func morph(m []bool, size int, erode bool) []bool {
	out := make([]bool, len(m))
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			v := erode
			for dr := -1; dr <= 1 && v == erode; dr++ {
				for dc := -1; dc <= 1; dc++ {
					rr, cc := r+dr, c+dc
					in := rr >= 0 && rr < size && cc >= 0 && cc < size && m[rr*size+cc]
					if erode && !in {
						v = false
						break
					}
					if !erode && in {
						v = true
						break
					}
				}
			}
			out[r*size+c] = v
		}
	}
	return out
}

// Patch: a connected group of changed pixels (8-connectivity).
type Patch struct {
	Pixels                 int
	SumRow, SumCol         int
	MinR, MinC, MaxR, MaxC int
}

// Components labels the connected groups of a mask; patches smaller than minPixels are dropped
// (their label stays 0). Labels start at 1.
func Components(m []bool, size, minPixels int) ([]int32, []Patch) {
	label := make([]int32, len(m))
	var patches []Patch
	stack := []int{}
	next := int32(1)
	for start := range m {
		if !m[start] || label[start] != 0 {
			continue
		}
		p := Patch{MinR: size, MinC: size, MaxR: -1, MaxC: -1}
		members := []int{}
		label[start] = next
		stack = append(stack[:0], start)
		for len(stack) > 0 {
			i := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			members = append(members, i)
			r, c := i/size, i%size
			p.Pixels++
			p.SumRow += r
			p.SumCol += c
			p.MinR, p.MaxR = min(p.MinR, r), max(p.MaxR, r)
			p.MinC, p.MaxC = min(p.MinC, c), max(p.MaxC, c)
			for dr := -1; dr <= 1; dr++ {
				for dc := -1; dc <= 1; dc++ {
					rr, cc := r+dr, c+dc
					if rr < 0 || rr >= size || cc < 0 || cc >= size {
						continue
					}
					j := rr*size + cc
					if m[j] && label[j] == 0 {
						label[j] = next
						stack = append(stack, j)
					}
				}
			}
		}
		if p.Pixels < minPixels {
			for _, i := range members {
				label[i] = -1
			}
			continue
		}
		patches = append(patches, p)
		next++
	}
	for i, l := range label {
		if l < 0 {
			label[i] = 0
		}
	}
	return label, patches
}
