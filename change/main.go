// canopy-change: "what changed between two dates" over the land-cover maps of the tile server.
//
//	canopy-change [-tiles http://127.0.0.1:8710] [-port 8720] [-static ../web/dist]
//
//	GET /api/regions                                 the tile server's regions
//	GET /api/change?region=pipera&from=2019&to=2025[&min_area=50][&dnbr=0.1]
//	    totals per kind of change (hectares), the full transition matrix, and the patches as
//	    GeoJSON (outline = bounding box, centroid, area). Deforestation must also show a drop
//	    of the normalised burn ratio of at least dnbr (0 turns the check off).
//	GET /api/change.png?region=...&from=...&to=...   the patches as a transparent overlay
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"log"
	"math"
	"net/http"
	"strconv"
	"sync"
	"time"
)

type Region struct {
	Name    string        `json:"name"`
	Split   string        `json:"split"`
	EPSG    int           `json:"epsg"`
	Left    float64       `json:"left"`
	Top     float64       `json:"top"`
	Size    int           `json:"size"`
	Years   []int         `json:"years"`
	Corners [4][2]float64 `json:"corners"`
}

type Server struct {
	tiles  string
	client *http.Client
	mu     sync.Mutex
	maps   map[string][]uint8
	idx    map[string][]int16
}

func (s *Server) regions() ([]Region, error) {
	resp, err := s.client.Get(s.tiles + "/regions")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var rs []Region
	return rs, json.NewDecoder(resp.Body).Decode(&rs)
}

func (s *Server) region(name string) (Region, error) {
	rs, err := s.regions()
	if err != nil {
		return Region{}, err
	}
	for _, r := range rs {
		if r.Name == name {
			return r, nil
		}
	}
	return Region{}, fmt.Errorf("no region %q", name)
}

// classMap fetches (and keeps) a region's land-cover map for a year.
func (s *Server) classMap(r Region, year int) ([]uint8, error) {
	key := fmt.Sprintf("%s/%d", r.Name, year)
	s.mu.Lock()
	m, ok := s.maps[key]
	s.mu.Unlock()
	if ok {
		return m, nil
	}
	resp, err := s.client.Get(fmt.Sprintf("%s/classes/%s/%d", s.tiles, r.Name, year))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("tile server: %s", bytes.TrimSpace(body))
	}
	if len(body) != r.Size*r.Size {
		return nil, fmt.Errorf("class map of %d bytes for a %d x %d grid", len(body), r.Size, r.Size)
	}
	s.mu.Lock()
	s.maps[key] = body
	s.mu.Unlock()
	return body, nil
}

// index fetches (and keeps) a summer spectral index of a region.
func (s *Server) index(r Region, year int, name string) ([]int16, error) {
	key := fmt.Sprintf("%s/%d/%s", r.Name, year, name)
	s.mu.Lock()
	v, ok := s.idx[key]
	s.mu.Unlock()
	if ok {
		return v, nil
	}
	resp, err := s.client.Get(fmt.Sprintf("%s/index/%s/%d/%s", s.tiles, r.Name, year, name))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("tile server: %s", bytes.TrimSpace(body))
	}
	if len(body) != 2*r.Size*r.Size {
		return nil, fmt.Errorf("index of %d bytes for a %d x %d grid", len(body), r.Size, r.Size)
	}
	v = make([]int16, r.Size*r.Size)
	for i := range v {
		v[i] = int16(uint16(body[2*i]) | uint16(body[2*i+1])<<8)
	}
	s.mu.Lock()
	s.idx[key] = v
	s.mu.Unlock()
	return v, nil
}

type Result struct {
	Region      string                  `json:"region"`
	From        int                     `json:"from"`
	To          int                     `json:"to"`
	MinPixels   int                     `json:"min_pixels"`
	DNBR        float64                 `json:"dnbr_min"`
	PixelM2     float64                 `json:"pixel_m2"`
	Totals      map[string]float64      `json:"totals_ha"`
	Transitions [nClasses][nClasses]int `json:"transitions"`
	Patches     map[string]any          `json:"patches"`
	Overlay     map[string]any          `json:"overlay"`
	labels      map[string][]int32
}

func (s *Server) change(name string, from, to, minPixels int, dnbr float64) (*Result, error) {
	r, err := s.region(name)
	if err != nil {
		return nil, err
	}
	a, err := s.classMap(r, from)
	if err != nil {
		return nil, err
	}
	b, err := s.classMap(r, to)
	if err != nil {
		return nil, err
	}
	u, ok := utmFromEPSG(r.EPSG)
	if !ok {
		return nil, fmt.Errorf("EPSG:%d is not UTM", r.EPSG)
	}
	toLonLat := func(row, col float64) [2]float64 {
		lon, lat := u.inverse(r.Left+10*col, r.Top-10*row)
		return [2]float64{lon, lat}
	}
	var nbrA, nbrB []int16
	if dnbr > 0 {
		if nbrA, err = s.index(r, from, "nbr"); err != nil {
			return nil, err
		}
		if nbrB, err = s.index(r, to, "nbr"); err != nil {
			return nil, err
		}
	}
	res := &Result{Region: name, From: from, To: to, MinPixels: minPixels, DNBR: dnbr, PixelM2: 100, Totals: map[string]float64{},
		Transitions: Transitions(a, b), labels: map[string][]int32{}}
	var features []map[string]any
	for _, k := range Kinds {
		m := Mask(a, b, k)
		if k.Name == "deforestation" && dnbr > 0 {
			m = Confirm(m, nbrA, nbrB, int(math.Round(dnbr*10000)))
		}
		lab, patches := Components(Open(m, r.Size), r.Size, minPixels)
		res.labels[k.Name] = lab
		total := 0
		for i, p := range patches {
			total += p.Pixels
			ring := [][2]float64{
				toLonLat(float64(p.MinR), float64(p.MinC)), toLonLat(float64(p.MinR), float64(p.MaxC+1)),
				toLonLat(float64(p.MaxR+1), float64(p.MaxC+1)), toLonLat(float64(p.MaxR+1), float64(p.MinC)),
				toLonLat(float64(p.MinR), float64(p.MinC)),
			}
			features = append(features, map[string]any{
				"type":     "Feature",
				"geometry": map[string]any{"type": "Polygon", "coordinates": [][][2]float64{ring}},
				"properties": map[string]any{
					"id": fmt.Sprintf("%s-%d", k.Name, i+1), "kind": k.Name, "pixels": p.Pixels,
					"area_ha":  float64(p.Pixels) * res.PixelM2 / 1e4,
					"centroid": toLonLat(float64(p.SumRow)/float64(p.Pixels)+0.5, float64(p.SumCol)/float64(p.Pixels)+0.5),
				},
			})
		}
		res.Totals[k.Name] = float64(total) * res.PixelM2 / 1e4
	}
	if features == nil {
		features = []map[string]any{}
	}
	res.Patches = map[string]any{"type": "FeatureCollection", "features": features}
	res.Overlay = map[string]any{
		"url":     fmt.Sprintf("/api/change.png?region=%s&from=%d&to=%d&min_area=%d&dnbr=%g", name, from, to, minPixels, dnbr),
		"corners": r.Corners,
	}
	return res, nil
}

var kindColor = map[string]color.NRGBA{"deforestation": {255, 140, 0, 230}, "construction": {230, 0, 90, 230}}

func (res *Result) overlay(size int) []byte {
	img := image.NewNRGBA(image.Rect(0, 0, size, size))
	for _, k := range Kinds {
		for i, l := range res.labels[k.Name] {
			if l > 0 {
				img.SetNRGBA(i%size, i/size, kindColor[k.Name])
			}
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
}

func params(r *http.Request) (string, int, int, int, float64, error) {
	q := r.URL.Query()
	from, err1 := strconv.Atoi(q.Get("from"))
	to, err2 := strconv.Atoi(q.Get("to"))
	if q.Get("region") == "" || err1 != nil || err2 != nil {
		return "", 0, 0, 0, 0, fmt.Errorf("need region, from and to")
	}
	minPixels := 50 // 0.5 ha
	if v := q.Get("min_area"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			return "", 0, 0, 0, 0, fmt.Errorf("min_area: a number of pixels")
		}
		minPixels = n
	}
	dnbr := 0.1
	if v := q.Get("dnbr"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || f < 0 {
			return "", 0, 0, 0, 0, fmt.Errorf("dnbr: a drop of the burn ratio, 0 to turn off")
		}
		dnbr = f
	}
	return q.Get("region"), from, to, minPixels, dnbr, nil
}

func main() {
	tiles := flag.String("tiles", "http://127.0.0.1:8710", "tile server")
	port := flag.Int("port", 8720, "port")
	static := flag.String("static", "", "serve the built map from this directory")
	flag.Parse()
	s := &Server{tiles: *tiles, client: &http.Client{Timeout: 10 * time.Minute}, maps: map[string][]uint8{}, idx: map[string][]int16{}}
	mux := http.NewServeMux()
	writeJSON := func(w http.ResponseWriter, v any) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(v)
	}
	mux.HandleFunc("GET /api/regions", func(w http.ResponseWriter, r *http.Request) {
		rs, err := s.regions()
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		writeJSON(w, rs)
	})
	mux.HandleFunc("GET /api/change", func(w http.ResponseWriter, r *http.Request) {
		name, from, to, minPixels, dnbr, err := params(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		res, err := s.change(name, from, to, minPixels, dnbr)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		writeJSON(w, res)
	})
	mux.HandleFunc("GET /api/change.png", func(w http.ResponseWriter, r *http.Request) {
		name, from, to, minPixels, dnbr, err := params(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		res, err := s.change(name, from, to, minPixels, dnbr)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		rg, _ := s.region(name)
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(res.overlay(rg.Size))
	})
	if *static != "" {
		mux.Handle("/", http.FileServer(http.Dir(*static)))
	}
	addr := fmt.Sprintf("127.0.0.1:%d", *port)
	log.Printf("listening on http://%s (tiles from %s)", addr, *tiles)
	log.Fatal(http.ListenAndServe(addr, cors(mux)))
}

func cors(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		h.ServeHTTP(w, r)
	})
}
