package gateway

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

type probeCounts struct {
	mu     sync.Mutex
	values map[string]int
}

func (c *probeCounts) hit(path string)       { c.mu.Lock(); defer c.mu.Unlock(); c.values[path]++ }
func (c *probeCounts) count(path string) int { c.mu.Lock(); defer c.mu.Unlock(); return c.values[path] }
func (c *probeCounts) String() string        { c.mu.Lock(); defer c.mu.Unlock(); return fmt.Sprint(c.values) }

// #5022's wording puts a model id between "model" and "does not exist".
// The request id below is a fixture value.
const probeModelRefusal = "onehub: The model `ZHIPU/GLM-5.3` does not exist or you do not have access to it. (request id: 20261106123000)"

const probeGood = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"probe ok\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"status\":\"completed\"}}\n\n"

func TestAmbiguousProtocolProbeGuards(t *testing.T) {
	for _, scope := range []string{"multi-key", "key-protocol", "pass-errors", "explicit-api", "sealed", "quota", "auth", "policy"} {
		t.Run(scope, func(t *testing.T) {
			t.Setenv("XDG_CONFIG_HOME", t.TempDir())
			t.Setenv("XDG_CACHE_HOME", t.TempDir())
			calls := &probeCounts{values: map[string]int{}}
			refusal := probeModelRefusal
			switch scope {
			case "quota":
				refusal += "; quota exhausted"
			case "auth":
				refusal += "; invalid API key"
			case "policy":
				refusal += "; content policy violation"
			}
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				io.Copy(io.Discard, r.Body)
				calls.hit(r.URL.Path)
				if r.URL.Path == "/v1/responses" {
					w.Header().Set("Content-Type", "text/event-stream")
					io.WriteString(w, probeGood)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(404)
				io.WriteString(w, `{"error":{"message":"`+refusal+`"}}`)
			}))
			t.Cleanup(up.Close)
			p := provider.Provider{ID: "probe", Name: "Probe", Key: "k", Models: []string{"probe-model"}, Chat: up.URL + "/v1", Responses: up.URL + "/v1"}
			body := `{"model":"probe-model","messages":[{"role":"user","content":"hello"}]}`
			switch scope {
			case "multi-key":
				p.Keys = []provider.KeyAccount{{Key: "second"}}
			case "key-protocol":
				p.KeyProtocol = provider.Chat
			case "pass-errors":
				p.PassErrors = true
			case "sealed":
				body = `{"model":"probe-model","messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"old","signature":"sealed"}]},{"role":"user","content":"hello"}]}`
			}
			if err := provider.Save(p); err != nil {
				t.Fatal(err)
			}
			if scope == "explicit-api" {
				if err := provider.SetModelAPI("probe/probe-model", "chat"); err != nil {
					t.Fatal(err)
				}
			}
			rec := httptest.NewRecorder()
			New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
			if rec.Code != 404 || calls.count("/v1/responses") != 0 {
				t.Fatalf("guard bypassed: status=%d body=%s calls=%v", rec.Code, rec.Body, calls)
			}
			// The behavior check above is intentionally independent of the new
			// provider learning API, so it can run against the baseline too.
		})
	}
}

func TestAmbiguousProtocolProbe(t *testing.T) {
	for _, refusal := range []string{probeModelRefusal, "url error, please check url！"} {
		for _, path := range []string{"/v1/chat/completions", "/v1/messages"} {
			t.Run(refusal+path, func(t *testing.T) {
				t.Setenv("XDG_CONFIG_HOME", t.TempDir())
				t.Setenv("XDG_CACHE_HOME", t.TempDir())
				calls := &probeCounts{values: map[string]int{}}
				up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					io.Copy(io.Discard, r.Body)
					calls.hit(r.URL.Path)
					if r.URL.Path == "/v1/responses" {
						w.Header().Set("Content-Type", "text/event-stream")
						w.Header().Set("X-Request-Id", "success-id")
						io.WriteString(w, probeGood)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(404)
					io.WriteString(w, `{"error":{"message":"`+refusal+`"}}`)
				}))
				t.Cleanup(up.Close)
				if err := provider.Save(provider.Provider{ID: "probe", Name: "Probe", Key: "k", Models: []string{"probe-model"}, Chat: up.URL + "/v1", Responses: up.URL + "/v1"}); err != nil {
					t.Fatal(err)
				}
				s := New()
				h := s.Handler()
				for i := 0; i < 2; i++ {
					rec := httptest.NewRecorder()
					body := fmt.Sprintf(`{"model":"probe-model","max_tokens":100,"stream":%t,"messages":[{"role":"user","content":"hello"}]}`, i == 0)
					h.ServeHTTP(rec, httptest.NewRequest("POST", path, strings.NewReader(body)))
					if rec.Code != 200 || !strings.Contains(rec.Body.String(), "probe ok") {
						t.Fatalf("try %d: status=%d body=%s calls=%v", i, rec.Code, rec.Body, calls)
					}
					if got := rec.Header().Get("X-Request-Id"); got != "success-id" {
						t.Fatalf("try %d: successful reply request id=%q", i, got)
					}
				}
				if calls.count("/v1/chat/completions") != 1 || calls.count("/v1/responses") != 2 {
					t.Fatalf("not bounded/learned: %v", calls)
				}
				if !s.fits("probe", "probe-model", provider.Chat) {
					t.Fatal("ambiguous rejection polluted shared unfit")
				}
			})
		}
	}
}

func TestAmbiguousProtocolProbeLearnedCanChange(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	var changed atomic.Bool
	calls := &probeCounts{values: map[string]int{}}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		calls.hit(r.URL.Path)
		if !changed.Load() && r.URL.Path == "/v1/responses" {
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, probeGood)
			return
		}
		if changed.Load() && r.URL.Path == "/v1/chat/completions" {
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"chat recovered\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(404)
		io.WriteString(w, `{"error":{"message":"`+probeModelRefusal+`"}}`)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{ID: "probe", Name: "Probe", Key: "k", Models: []string{"probe-model"}, Chat: up.URL + "/v1", Responses: up.URL + "/v1"}); err != nil {
		t.Fatal(err)
	}
	s := New()
	for i, want := range []string{"probe ok", "chat recovered", "chat recovered"} {
		changed.Store(i > 0)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"probe-model","messages":[{"role":"user","content":"hello"}]}`)))
		if rec.Code != 200 || !strings.Contains(rec.Body.String(), want) {
			t.Fatalf("try %d: status=%d body=%s calls=%v", i, rec.Code, rec.Body, calls)
		}
	}
	if calls.count("/v1/responses") != 2 || calls.count("/v1/chat/completions") != 3 {
		t.Fatalf("learned preference locked protocol or looped: %v", calls)
	}
}

func TestAmbiguousProtocolProbeRestoresRefusal(t *testing.T) {
	for _, reply := range []string{
		`{"error":{"message":"still refused"}}`,
		"data: [DONE]\n\n",
		"event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"bad\"}}\n\n",
		"<html>not an API</html>",
		"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"cut off\"}\n\n",
	} {
		t.Run(reply, func(t *testing.T) {
			t.Setenv("XDG_CONFIG_HOME", t.TempDir())
			t.Setenv("XDG_CACHE_HOME", t.TempDir())
			calls := &probeCounts{values: map[string]int{}}
			const original = `{"error":{"message":"` + probeModelRefusal + `"}}`
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				io.Copy(io.Discard, r.Body)
				calls.hit(r.URL.Path)
				if r.URL.Path == "/v1/chat/completions" {
					w.Header().Set("Content-Type", "application/json")
					w.Header().Set("X-Request-Id", "original-id")
					w.WriteHeader(404)
					io.WriteString(w, original)
				} else {
					w.Header().Set("X-Request-Id", "probe-id")
					w.Header().Set("Content-Type", "text/event-stream")
					io.WriteString(w, reply)
				}
			}))
			t.Cleanup(up.Close)
			if err := provider.Save(provider.Provider{ID: "probe", Name: "Probe", Key: "k", Models: []string{"probe-model"}, Chat: up.URL + "/v1", Responses: up.URL + "/v1"}); err != nil {
				t.Fatal(err)
			}
			s := New()
			for i := 0; i < 2; i++ {
				rec := httptest.NewRecorder()
				s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"probe-model","messages":[{"role":"user","content":"hello"}]}`)))
				if rec.Code != 404 || !strings.Contains(rec.Body.String(), "does not exist") || strings.Contains(rec.Body.String(), "cut off") {
					t.Fatalf("original lost: status=%d body=%s", rec.Code, rec.Body)
				}
				if rec.Header().Get("X-Request-Id") != "original-id" {
					t.Fatalf("request id=%q", rec.Header().Get("X-Request-Id"))
				}
			}
			if calls.count("/v1/chat/completions") != 2 || calls.count("/v1/responses") != 2 {
				t.Fatalf("failed probe was learned or not tried: %v", calls)
			}
			if !s.fits("probe", "probe-model", provider.Chat) || !s.fits("probe", "probe-model", provider.Responses) {
				t.Fatal("failed probe polluted unfit")
			}
		})
	}
}
