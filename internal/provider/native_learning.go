package provider

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/settings"
)

// Success has a soft refresh deadline: a model that worked stays available
// while its configuration is unchanged, even if its next verification fails.
// Failures only cool probing. Neither kind of evidence is a user preference.
const learnedAPITTL = 30 * time.Minute
const failedAPITTL = time.Minute
const learnedAPILimit = 4096

type learnedAPI struct {
	Protocol Protocol
	Until    time.Time
	Provider string
	file     string
}

var learnedAPIs = struct {
	sync.Mutex
	entries    map[[32]byte]learnedAPI
	scopes     map[string]bool
	loaded     map[string]bool
	unreadable map[string]bool
}{entries: make(map[[32]byte]learnedAPI), scopes: make(map[string]bool), loaded: make(map[string]bool), unreadable: make(map[string]bool)}

// learnedCount and learnedReady are read for every model of every list: a
// magpie that has verified nothing answers from these two words alone, with
// no path joined, no lock taken and nothing allocated. The count is the
// successes in memory; ready says the file was read once.
var learnedCount atomic.Int64
var learnedReady atomic.Bool

// Disk work is serialized outside the cache lock its readers need.
var learnedAPIIO sync.Mutex

// learnedPath is the file the successes are kept in, once per configuration
// folder: the path is joined on every look at it, and a look at it is every
// model of every list.
var learnedPath struct {
	sync.Mutex
	by map[string]string
}

func learnedAPIPath() string {
	dir := learnedConfigDir()
	learnedPath.Lock()
	defer learnedPath.Unlock()
	if learnedPath.by == nil {
		learnedPath.by = map[string]string{}
	}
	if p, ok := learnedPath.by[dir]; ok {
		return p
	}
	p := filepath.Join(dir, "native-api-cache.json")
	learnedPath.by[dir] = p
	return p
}

// learnedConfigDir is the folder magpie keeps its files in, once per path:
// every model of every list asks for it.
var learnedDir struct {
	sync.Mutex
	by map[string]string
}

func learnedConfigDir() string {
	path := Path()
	learnedDir.Lock()
	defer learnedDir.Unlock()
	if learnedDir.by == nil {
		learnedDir.by = map[string]string{}
	}
	if d, ok := learnedDir.by[path]; ok {
		return d
	}
	d := filepath.Dir(path)
	learnedDir.by[path] = d
	return d
}

func loadLearnedAPIs() {
	if learnedReady.Load() {
		return
	}
	path := learnedAPIPath()
	learnedAPIs.Lock()
	loaded := learnedAPIs.loaded[path]
	learnedAPIs.Unlock()
	if loaded {
		learnedReady.Store(true)
		return
	}
	learnedAPIIO.Lock()
	defer learnedAPIIO.Unlock()
	learnedAPIs.Lock()
	loaded = learnedAPIs.loaded[path]
	learnedAPIs.Unlock()
	if loaded {
		return
	}
	var disk map[string]learnedAPI
	f, err := os.Open(path)
	if err == nil {
		err = json.NewDecoder(io.LimitReader(f, 2<<20)).Decode(&disk)
		f.Close()
	}
	unreadable := err != nil && !os.IsNotExist(err)
	learnedAPIs.Lock()
	defer learnedAPIs.Unlock()
	learnedAPIs.loaded[path] = true
	learnedAPIs.unreadable[path] = unreadable
	learnedReady.Store(true)
	if err != nil {
		return
	}
	for key, v := range disk {
		if len(learnedAPIs.entries) >= learnedAPILimit {
			break
		}
		b, err := hex.DecodeString(key)
		if err != nil || len(b) != sha256.Size || v.Provider == "" || !slices.Contains(Protocols, v.Protocol) {
			continue
		}
		var k [32]byte
		copy(k[:], b)
		v.file = path
		learnedAPIs.entries[k] = v
		learnedAPIs.scopes[v.Provider] = true
	}
	learnedCount.Store(int64(len(learnedAPIs.entries)))
}

func persistLearnedAPIs() {
	path := learnedAPIPath()
	learnedAPIIO.Lock()
	defer learnedAPIIO.Unlock()
	learnedAPIs.Lock()
	if learnedAPIs.unreadable[path] {
		learnedAPIs.Unlock()
		return
	}
	disk := make(map[string]learnedAPI)
	for k, v := range learnedAPIs.entries {
		if v.file == path && v.Protocol != "" {
			disk[hex.EncodeToString(k[:])] = v
		}
	}
	learnedAPIs.Unlock()
	b, err := json.Marshal(disk)
	if err != nil {
		return
	}
	if os.MkdirAll(filepath.Dir(path), 0o700) != nil {
		return
	}
	// This cache is replaceable evidence; a failed write leaves memory and
	// the last atomically stored successful snapshot intact.
	_ = writePrivate(path, append(b, '\n'))
}

// learningFingerprint captures request configuration and model mappings once
// for a batch. Its JSON prefix keeps existing persisted SHA-256 keys valid.
// Only the two model-specific strings are encoded and hashed per lookup.
type learningFingerprint struct {
	prefix []byte
	pid    string
	ids    localIDs
	wires  map[string]string
}

func (p Provider) learningFingerprint() (learningFingerprint, bool) {
	if p.Account != nil || p.Preset != "" || p.Off || p.ServesNone() {
		return learningFingerprint{}, false
	}
	config := Provider{ID: p.ID, Key: p.Key, KeyProtocol: p.KeyProtocol, KeyPlan: p.KeyPlan,
		Chat: p.Chat, Responses: p.Responses, Anthropic: p.Anthropic, Gemini: p.Gemini,
		BaseAPI: p.BaseAPI, NativeInferred: p.NativeInferred, Headers: p.Headers,
		ExtraBody: p.ExtraBody, PassHeaders: p.PassHeaders, AuthHeader: p.AuthHeader, Proxy: p.ProxyChoice(),
		InsecureTLS: p.InsecureTLS, HTTP1: p.HTTP1, NativeCompaction: p.NativeCompaction}
	prefix, err := json.Marshal(struct {
		Path   string
		Config Provider
	}{Path(), config})
	if err != nil {
		return learningFingerprint{}, false
	}
	s := settings.Shared()
	return learningFingerprint{prefix: prefix[:len(prefix)-1], pid: p.ID, ids: idsIn(s.ModelIDs, p.ID), wires: s.Wires()}, true
}

func (f learningFingerprint) key(model string) ([32]byte, bool) {
	own := f.ids.ownOf(model)
	suffix, err := json.Marshal(struct{ Model, Wire string }{own, UpstreamNameAsIn(f.wires, f.pid, model, own)})
	if err != nil {
		return [32]byte{}, false
	}
	h := sha256.New()
	h.Write(f.prefix)
	h.Write([]byte{','})
	h.Write(suffix[1:])
	var key [32]byte
	h.Sum(key[:0])
	return key, true
}

func (p Provider) learningKey(model string) ([32]byte, bool) {
	f, ok := p.learningFingerprint()
	if !ok {
		return [32]byte{}, false
	}
	return f.key(model)
}

func (p Provider) learned(model string) (learnedAPI, bool) {
	loadLearnedAPIs()
	if !p.hasEvidence() {
		return learnedAPI{}, false
	}
	key, ok := p.learningKey(model)
	if !ok {
		return learnedAPI{}, false
	}
	learnedAPIs.Lock()
	v, ok := learnedAPIs.entries[key]
	// Successful evidence expires only when its configuration changes. The
	// deadline asks the next Fetch to refresh it, without dropping a route.
	if ok && v.Protocol == "" && time.Now().After(v.Until) {
		delete(learnedAPIs.entries, key)
		ok = false
	}
	learnedAPIs.Unlock()
	return v, ok
}

// hasEvidence is whether this provider may have any remembered success at
// all: a magpie that has verified nothing answers before a path is joined or
// a lock is taken, and a provider with nothing in memory pays one scope
// look-up for each of its models.
func (p Provider) hasEvidence() bool {
	if learnedCount.Load() == 0 {
		return false
	}
	learnedAPIs.Lock()
	has := learnedAPIs.scopes[p.ID]
	learnedAPIs.Unlock()
	return has
}

// LearnedAPI returns this model's successful API for these effective endpoints
// and credentials. It is an automatic preference, not an API lock. A stale
// success stays usable until a refreshed verification finds a replacement.
func (p Provider) LearnedAPI(model string) (Protocol, bool) {
	return p.LearnedAPIReader()(model)
}

func noLearnedAPI(string) (Protocol, bool) { return "", false }

// LearnedAPIReader captures the configuration and model mappings for a batch
// of lookups. Create a new reader after settings change. An owner with no
// evidence pays for its path/load/scope check once, not once per listed model.
func (p Provider) LearnedAPIReader() func(string) (Protocol, bool) {
	if p.Account != nil || p.Preset != "" || p.Off || p.ServesNone() {
		return noLearnedAPI
	}
	loadLearnedAPIs()
	if !p.hasEvidence() {
		return noLearnedAPI
	}
	f, ok := p.learningFingerprint()
	if !ok {
		return noLearnedAPI
	}
	speaks := p.Speaks()
	return func(model string) (Protocol, bool) {
		key, ok := f.key(model)
		if !ok {
			return "", false
		}
		learnedAPIs.Lock()
		v, ok := learnedAPIs.entries[key]
		learnedAPIs.Unlock()
		// Successful evidence has a soft deadline; negative cooldowns do
		// not become preferences, whether they have expired or not.
		if !ok || v.Protocol == "" || !slices.Contains(speaks, v.Protocol) {
			return "", false
		}
		return v.Protocol, true
	}
}

// RememberAPI records a validated successful model reply. Callers must not
// call it for an HTTP success carrying an error or an incomplete stream.
// It never changes a user's explicit model API preference.
func (p Provider) RememberAPI(model string, proto Protocol) bool {
	if model == "" || !slices.Contains(Protocols, proto) || !slices.Contains(p.Speaks(), proto) {
		return false
	}
	if _, off := ModelDisabled(p.ID, model); off {
		return false
	}
	if _, set := p.apiSet(model); set {
		return false
	}
	return p.rememberLearning(model, proto, learnedAPITTL)
}

func (p Provider) rememberLearning(model string, proto Protocol, ttl time.Duration) bool {
	loadLearnedAPIs()
	key, ok := p.learningKey(model)
	if !ok {
		return false
	}
	scope := p.ID
	learnedAPIs.Lock()
	now := time.Now()
	old, had := learnedAPIs.entries[key]
	changed := proto != "" && (!had || old.Protocol != proto)
	if len(learnedAPIs.entries) >= learnedAPILimit && !had {
		for k, v := range learnedAPIs.entries {
			if v.Protocol == "" && now.After(v.Until) {
				delete(learnedAPIs.entries, k)
			}
		}
		if len(learnedAPIs.entries) >= learnedAPILimit {
			for k, v := range learnedAPIs.entries {
				delete(learnedAPIs.entries, k)
				changed = changed || v.Protocol != ""
				break
			}
		}
	}
	if len(learnedAPIs.scopes) >= learnedAPILimit {
		clear(learnedAPIs.scopes)
		clear(learnedAPIs.entries)
		changed = true
	}
	learnedAPIs.scopes[scope] = true
	// A failed refresh never erases a previous success. It only delays the
	// next verification of the same model and configuration.
	if proto == "" && had && old.Protocol != "" {
		proto = old.Protocol
	}
	learnedAPIs.entries[key] = learnedAPI{Protocol: proto, Until: now.Add(ttl), Provider: p.ID, file: learnedAPIPath()}
	learnedCount.Store(int64(len(learnedAPIs.entries)))
	learnedAPIs.Unlock()
	if changed {
		Changed()
		catalog.Touched()
	}
	// A success is written to the file when it is new or its protocol
	// changed: that is what carries it over a restart. A renewal of the same
	// success only moves its deadline in memory, so a list that keeps
	// answering does not write the file on every refresh.
	if proto != "" && (!had || old.Protocol != proto) {
		persistLearnedAPIs()
	}
	return true
}

func (p Provider) inferredAPI(proto Protocol) bool {
	base := p.NativeInferred[proto]
	return base != "" && base == p.Base(proto)
}
